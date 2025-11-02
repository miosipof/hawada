# data/llms.py
"""
LLM dataset utilities: streaming packing, loaders, and vocab alignment.

Features
--------
- Iterable packed dataset for HF corpora (streaming or local).
- Deterministic worker sharding and optional buffered shuffle.
- BOS/EOS insertion and fixed-length packing (no padding inside packs).
- Config-friendly DataLoader builders (train/val).
- Optional teacher→student vocab-id remapping for KD.

Example (from a YAML-like dict):
    from data.llms import build_llm_dataloaders_from_cfg

    dl_train, dl_val, tok_student, tok_teacher, st_map = build_llm_dataloaders_from_cfg({
        "dataset_name": "fla-hub/slimpajama-test",
        "tokenizer_name": "meta-llama/Llama-3.2-1B-Instruct",
        "seq_len": 1024,
        "train": {"batch_size": 1, "num_workers": 4, "streaming": False, "size": 3000, "shuffle_buffer": 20000, "seed": 1234},
        "val":   {"batch_size": 1, "num_workers": 4, "streaming": False, "size": 300,  "shuffle_buffer": 20000, "seed": 1234},
    })
"""

from __future__ import annotations

import random
from typing import Iterator, Optional, Dict, Any, Tuple, List, Union

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

try:
    from datasets import load_dataset
except Exception as e:
    load_dataset = None

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except Exception:
    AutoTokenizer = None
    PreTrainedTokenizerBase = None

from transformers import AutoTokenizer

# ----------------------------
# Helpers
# ----------------------------
# data/llms.py

def _ensure_tokenizer(tokenizer_or_name, use_fast: bool = True):
    """
    Accepts either a tokenizer instance or a hub id (str).
    Always returns a HF tokenizer object with a valid pad token.
    """
    # 1) Load if a hub id was passed
    if isinstance(tokenizer_or_name, str):
        tok = AutoTokenizer.from_pretrained(tokenizer_or_name, use_fast=use_fast)
    else:
        tok = tokenizer_or_name  # already a tokenizer

    # 2) Ensure a pad token exists (common for causal LMs)
    if tok.pad_token_id is None:
        # Prefer EOS, then UNK, else create a PAD token
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.unk_token is not None:
            tok.pad_token = tok.unk_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})

    # 3) Guard insane model_max_length values some tokenizers ship with
    try:
        if getattr(tok, "model_max_length", None) and tok.model_max_length > 1_000_000:
            tok.model_max_length = 2048
    except Exception:
        pass

    return tok



def _extract_text(example: Dict[str, Any], prefer_field: Optional[str] = "text") -> str:
    if prefer_field and prefer_field in example and isinstance(example[prefer_field], str):
        txt = example[prefer_field]
    else:
        # fallback: join all string-like fields
        parts = []
        for k, v in example.items():
            if isinstance(v, str):
                parts.append(v)
        txt = "\n".join(parts)
    if not isinstance(txt, str):
        txt = str(txt)
    # ensure newline termination (helps packing boundaries)
    if not txt.endswith("\n"):
        txt += "\n"
    return txt


# ----------------------------
# Packed text iterable dataset
# ----------------------------
class PackedTextDataset(IterableDataset):
    """
    Streams one or multiple datasets and yields fixed-length token blocks.

    Each item: {"input_ids": Long[seq_len], "attention_mask": Long[seq_len]}.

    Arguments
    ---------
    datasets: List of dicts, each with:
        - dataset_name (required): HF dataset path or local builder
        - subset (split): default "train"
        - name (config name): optional
        - data_dir: optional
        - streaming: bool
        - size: int or None (materialized select[:size] when streaming=False)
        - text_field: str or None
    tokenizer / tokenizer_name: HF tokenizer or name.
    seq_len: int, pack size.
    add_bos / add_eos: bool.
    shuffle_buffer: int, only for streaming mode (HF shuffle).
    seed: base RNG seed for deterministic order.
    """

    def __init__(
        self,
        *,
        datasets: List[Dict[str, Any]],
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        tokenizer_name: Optional[str] = None,
        seq_len: int = 1024,
        add_bos: bool = False,
        add_eos: bool = True,
        shuffle_buffer: int = 20_000,
        seed: int = 42,
    ):
        super().__init__()
        assert len(datasets) >= 1, "Provide at least one dataset spec"
        self.seq_len = int(seq_len)
        self.add_bos = bool(add_bos)
        self.add_eos = bool(add_eos)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        self.ds_specs = datasets

        self.tok = _ensure_tokenizer(tokenizer or tokenizer_name)
        self.eos_id = self.tok.eos_token_id if self.tok.eos_token_id is not None else self.tok.pad_token_id
        self.bos_id = self.tok.bos_token_id

        assert load_dataset is not None, "datasets library is required"

        # Build internal list of HF iterable datasets (one per spec)
        self._hf_streams = []
        for spec in self.ds_specs:
            name = spec.get("dataset_name")
            subset = spec.get("subset", "train")
            ds_name = spec.get("name", None)
            data_dir = spec.get("data_dir", None)
            streaming = bool(spec.get("streaming", True))
            size = spec.get("size", None)

            ds = load_dataset(name, name=ds_name, data_dir=data_dir, split=subset, streaming=streaming)

            if not streaming and size is not None:
                # materialized dataset: select first size items
                ds = ds.select(range(int(size)))

            if streaming and self.shuffle_buffer > 0:
                ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

            self._hf_streams.append((ds, bool(streaming), spec.get("text_field", "text")))

    def _iter_source(self) -> Iterator[Dict]:
        """
        Round-robin across sources; shard deterministically by worker id.
        """
        # Prepare iterators for all sources
        iters = [iter(ds) for (ds, _, _) in self._hf_streams]
        active = list(range(len(iters)))
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        nw = wi.num_workers if wi is not None else 1
        idx = 0

        while active:
            i = active[idx % len(active)]
            try:
                ex = next(iters[i])
                # worker sharding: keep items whose global index % num_workers == wid
                # (Approximate sharding for streaming; ok to skip in materialized case too.)
                if (idx % nw) == wid:
                    yield (i, ex)  # include source id
                idx += 1
            except StopIteration:
                # drop exhausted source
                iters.pop(i)
                active.remove(i)
                # reindex remaining
                active = list(range(len(iters)))
                idx = 0
            except Exception:
                # skip broken example, continue
                idx += 1

    def _token_stream(self) -> Iterator[int]:
        """
        Concatenate tokens across documents with optional BOS/EOS.
        """
        # Seed per worker for deterministic token stream
        wi = get_worker_info()
        add_seed = (wi.id if wi is not None else 0)
        rng = random.Random(self.seed + add_seed)

        for src_id, ex in self._iter_source():
            text_field = self._hf_streams[src_id][2]
            text = _extract_text(ex, prefer_field=text_field)
            # add simple noise at doc boundaries (optional): here kept minimal
            ids = self.tok.encode(text, add_special_tokens=False)
            if self.add_bos and self.bos_id is not None:
                yield self.bos_id
            for tid in ids:
                yield tid
            if self.add_eos and self.eos_id is not None:
                yield self.eos_id

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buf: List[int] = []
        for tid in self._token_stream():
            buf.append(tid)
            while len(buf) >= self.seq_len:
                block = buf[: self.seq_len]
                buf = buf[self.seq_len :]
                ids = torch.tensor(block, dtype=torch.long)
                attn = torch.ones_like(ids, dtype=torch.long)
                yield {"input_ids": ids, "attention_mask": attn}


# ----------------------------
# Collate
# ----------------------------
def collate_packed(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Stack fixed-length blocks into a batch.
    """
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ----------------------------
# Builders
# ----------------------------
def build_llm_dataloader(
    *,
    dataset_name: str,
    tokenizer_name: str,
    seq_len: int = 1024,
    batch_size: int = 1,
    num_workers: int = 4,
    streaming: bool = True,
    size: Optional[int] = None,
    shuffle_buffer: int = 20_000,
    seed: int = 42,
    subset: str = "train",
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    text_field: Optional[str] = "text",
) -> Tuple[DataLoader, "PreTrainedTokenizerBase"]:
    """
    Convenience single-source builder matching your LLaMA examples.
    """
    tok = _ensure_tokenizer(tokenizer_name)
    ds = PackedTextDataset(
        datasets=[{
            "dataset_name": dataset_name,
            "subset": subset,
            "name": name,
            "data_dir": data_dir,
            "streaming": streaming,
            "size": size,
            "text_field": text_field,
        }],
        tokenizer=tok,
        seq_len=seq_len,
        add_bos=False,
        add_eos=True,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_packed,
    )
    return dl, tok


def build_llm_dataloaders_from_cfg(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, "PreTrainedTokenizerBase", Optional["PreTrainedTokenizerBase"], Optional[torch.Tensor]]:
    """
    Build (train_loader, val_loader, student_tokenizer, teacher_tokenizer, student_to_teacher_vocab_map)
    from a config dict like the 'data' section of your YAML recipe.

    Keys (examples):
      cfg = {
        "dataset_name": "fla-hub/slimpajama-test",
        "tokenizer_name": "meta-llama/Llama-3.2-1B-Instruct",
        "teacher_tokenizer_name": null | "meta-llama/Llama-3.2-1B-Instruct",
        "seq_len": 1024,
        "train": {...},
        "val":   {...},
      }
    """
    dataset_name = cfg.get("dataset_name")
    tokenizer_name = cfg.get("tokenizer_name")
    teacher_tok_name = cfg.get("teacher_tokenizer_name", None)
    seq_len = int(cfg.get("seq_len", 1024))

    # --- train ---
    tr = cfg.get("train", {})
    dl_train, tok_student = build_llm_dataloader(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        seq_len=seq_len,
        batch_size=int(tr.get("batch_size", 1)),
        num_workers=int(tr.get("num_workers", 4)),
        streaming=bool(tr.get("streaming", True)),
        size=tr.get("size", None),
        shuffle_buffer=int(tr.get("shuffle_buffer", 20000)),
        seed=int(tr.get("seed", 42)),
        subset=tr.get("subset", "train"),
        name=tr.get("name", None),
        data_dir=tr.get("data_dir", None),
        text_field=tr.get("text_field", "text"),
    )

    # --- val ---
    va = cfg.get("val", {})
    dl_val, tok_val = build_llm_dataloader(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        seq_len=seq_len,
        batch_size=int(va.get("batch_size", 1)),
        num_workers=int(va.get("num_workers", 4)),
        streaming=bool(va.get("streaming", True)),
        size=va.get("size", None),
        shuffle_buffer=int(va.get("shuffle_buffer", 20000)),
        seed=int(va.get("seed", 42)),
        subset=va.get("subset", "train"),
        name=va.get("name", None),
        data_dir=va.get("data_dir", None),
        text_field=va.get("text_field", "text"),
    )

    # optional teacher tokenizer (for KD vocab alignment)
    tok_teacher = None
    st_map = None
    if teacher_tok_name:
        tok_teacher = _ensure_tokenizer(teacher_tok_name)
        st_map = build_student_to_teacher_vocab_map(tok_student, tok_teacher)  # [Vt] mapping to student ids

    return dl_train, dl_val, tok_student, tok_teacher, st_map


# ----------------------------
# Student→Teacher vocab mapping
# ----------------------------
def build_student_to_teacher_vocab_map(
    student_tok: "PreTrainedTokenizerBase",
    teacher_tok: "PreTrainedTokenizerBase",
    *,
    unk_fallback: Optional[int] = None,
) -> torch.Tensor:
    """
    Create a tensor M of shape [V_teacher], where M[t_id] = corresponding student_id
    for the *same token string*. If a token string isn't in student vocab, map to
    `unk_fallback` (defaults to student's unk or eos if available, else 0).

    This is a pragmatic alignment for KL logits when tokenizers are (mostly) shared.
    """
    inv_student = {tok: idx for tok, idx in student_tok.get_vocab().items()}
    t_vocab = teacher_tok.get_vocab()
    if unk_fallback is None:
        unk_fallback = (
            student_tok.unk_token_id
            if student_tok.unk_token_id is not None
            else (student_tok.eos_token_id if student_tok.eos_token_id is not None else 0)
        )

    mapping = []
    for tok_str, t_id in sorted(t_vocab.items(), key=lambda kv: kv[1]):
        s_id = inv_student.get(tok_str, unk_fallback)
        mapping.append(int(s_id))
    return torch.tensor(mapping, dtype=torch.long)
