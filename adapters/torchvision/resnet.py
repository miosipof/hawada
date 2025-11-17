"""Torchvision ResNet adapter for hardware-aware pruning.

This adapter mirrors the ViT flow:
- Attaches grouped gates to BatchNorm2d outputs (one logit per group).
- Provides export_keepall (drop gates) and export_pruned (slice conv/bn/fc).
- Supplies param groups and a lightweight BN recalibration utility.

It targets torchvision-style ResNets (resnet18/34/50/101) with BasicBlock/Bottleneck.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import copy

import torch
import torch.nn as nn

from core.gates import GroupGate
from core.export import Rounding as CoreRounding, ExportPolicy as CoreExportPolicy
from core.utils import deepcopy_eval_cpu


# ----------------------------- Gate wrapper -----------------------------

class BNWithGate(nn.Module):
    """BatchNorm2d + GroupGate wrapper.

    The gate owns `logits`, `tau`, and `group_size`. Forward multiplies the BN
    output by a per-channel mask (grouped STE during training, hard/prob in eval).
    """
    def __init__(self, bn: nn.BatchNorm2d, group_size: int = 16, tau: float = 1.5, init_logit: float = 3.0,
                 hard_eval: bool = True):
        super().__init__()
        assert isinstance(bn, nn.BatchNorm2d)
        self.bn = bn
        C = int(bn.num_features)
        assert C % group_size == 0, f"BN channels {C} must be divisible by group_size={group_size}"
        self.gate = GroupGate(num_groups=C // group_size, group_size=group_size,
                              tau=tau, init_logit=init_logit, hard_during_eval=hard_eval)

    # Friendly accessors (used by proxy/export code)
    @property
    def logits(self):
        return self.gate.logits

    @property
    def tau(self):
        return self.gate.tau

    @property
    def group_size(self):
        return self.gate.group_size

    def kept_channels_soft(self) -> torch.Tensor:
        # expected kept channels = sum(sigmoid(logits/tau)) * group_size
        probs = torch.sigmoid(self.gate.logits / float(self.gate.tau))
        return probs.sum() * self.gate.group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn(x)
        m = self.gate.mask(self.training)  # [C]
        return y * m.view(1, -1, 1, 1)


# ----------------------------- Export policy -----------------------------

@dataclass
class ResNetExportPolicy:
    """Per-network export policy for ResNet.

    - `rounding.multiple_groups` applies to BN groups (channels snap to
      multiple_groups * group_size).
    - `min_keep_ratio` is a soft safety floor during export.
    - `warmup_steps` keeps all channels when step < warmup_steps.
    """
    warmup_steps: int = 0
    rounding: CoreRounding = CoreRounding(floor_groups=1, multiple_groups=1, min_keep_ratio=0.0)
    min_keep_ratio: float = 0.0


# ----------------------------- Utilities -----------------------------

@torch.no_grad()
def _slice_conv2d(conv: nn.Conv2d, keep_in: Optional[torch.Tensor] = None,
                  keep_out: Optional[torch.Tensor] = None) -> nn.Conv2d:
    W = conv.weight.data
    b = conv.bias.data if conv.bias is not None else None
    if keep_out is not None:
        W = W[keep_out]
        if b is not None:
            b = b[keep_out]
    if keep_in is not None:
        W = W[:, keep_in]
    out_c, in_c, kh, kw = W.shape
    new = nn.Conv2d(in_c, out_c, kernel_size=conv.kernel_size, stride=conv.stride,
                    padding=conv.padding, dilation=conv.dilation, groups=1,
                    bias=(b is not None), padding_mode=conv.padding_mode)
    new.weight.copy_(W)
    if b is not None:
        new.bias.copy_(b)
    return new

@torch.no_grad()
def _slice_bn2d(bn: nn.BatchNorm2d, keep_idx: torch.Tensor) -> nn.BatchNorm2d:
    new = nn.BatchNorm2d(int(keep_idx.numel()), eps=bn.eps, momentum=bn.momentum,
                         affine=True, track_running_stats=True)
    new.weight.copy_(bn.weight.data[keep_idx])
    new.bias.copy_(bn.bias.data[keep_idx])
    new.running_mean.copy_(bn.running_mean.data[keep_idx])
    new.running_var.copy_(bn.running_var.data[keep_idx])
    return new

# @torch.no_grad()
# def _choose_group_indices(gate_like: BNWithGate, policy: ResNetExportPolicy, step: int) -> torch.Tensor:
#     """Return sorted indices of kept *groups* for a BNWithGate.

#     Applies warmup, then rounds the estimated kept-group count using `policy.rounding`.
#     """
#     C = int(gate_like.bn.num_features)
#     G = int(C // gate_like.group_size)
#     if step < int(policy.warmup_steps):
#         return torch.arange(G)

#     # soft expected kept groups
#     probs = torch.sigmoid(gate_like.logits.float() / float(gate_like.tau))
#     k_est = int(round(float(probs.sum().cpu())))

#     # floors
#     min_groups = max(1, int(max(policy.min_keep_ratio, float(policy.rounding.min_keep_ratio)) * G))
#     k = max(min_groups, min(k_est, G))

#     # snap to multiple_groups (in group units)
#     M = int(max(1, policy.rounding.multiple_groups))
#     if M > 1:
#         k = max(M, (k // M) * M)  # floor to multiple of M
#         k = min(k, G)

#     # pick top-k groups by logits
#     grp = torch.topk(gate_like.logits.detach().float().cpu(), k, largest=True).indices.sort().values

#     return grp    

def _choose_group_indices(gbn, policy, *, force_groups=None, tag=""):
    C  = gbn.bn.num_features
    gs = getattr(gbn, "group_size", getattr(gbn, "group", None))
    assert gs and gs > 0, f"[{tag}] invalid group_size"
    G  = C // gs

    logits = gbn.logits.detach().float().cpu()
    tau    = float(getattr(gbn, "tau", 1.5))

    if force_groups is not None:
        # ✅ EXACT force: keep exactly this many groups (bounded by [1, G]).
        k = max(1, min(int(force_groups), int(G)))
    else:
        # original estimation path
        probs = torch.sigmoid(logits / tau)
        k_est = int(round(float(probs.sum().item())))
        floor_groups = max(1, int(float(getattr(policy, "min_keep_ratio", 0.0)) * G))
        k = max(floor_groups, min(k_est, int(G)))

        # apply multiple-of-groups snapping ONLY when NOT forcing
        mult = getattr(getattr(policy, "rounding", None), "multiple_groups", 1)
        if mult and mult > 1:
            k = (k // int(mult)) * int(mult)
            k = max(floor_groups, min(k, int(G)))

    grp_idx = torch.topk(logits, k, largest=True).indices.sort().values
    ch = torch.cat([torch.arange(int(i)*gs, (int(i)+1)*gs) for i in grp_idx]).long()
    # (optional assert to catch regressions early)
    assert ch.numel() == k * gs and ch.numel() <= C, f"[{tag}] channel idx size mismatch"
    return ch


# ----------------------------- Adapter -----------------------------

class ResNetAdapter:
    SUPPORTED = {"resnet18", "resnet34", "resnet50", "resnet101"}

    @staticmethod
    def _make_backbone(name: str = "resnet18", pretrained: bool = True, num_classes: int = 1000) -> nn.Module:
        from torchvision import models
        assert name in ResNetAdapter.SUPPORTED, f"Unknown backbone {name}"
        if name == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif name == "resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        elif name == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        else:
            base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None)
        if num_classes != 1000:
            base.fc = nn.Linear(base.fc.in_features, num_classes)
        return base

    # ---- attach gates ----
    @staticmethod
    def attach_gates(model: Optional[nn.Module] = None, *,
                     name: str = "resnet18", pretrained: bool = True,
                     group_size: int = 16, tau: float = 1.5, init_logit: float = 3.0,
                     hard_eval: bool = True, num_classes: int = 1000) -> nn.Module:
        m = model or ResNetAdapter._make_backbone(name, pretrained=pretrained, num_classes=num_classes)

        # Replace all BatchNorm2d modules with BNWithGate
        def _wrap_bn(mod: nn.Module):
            for child_name, child in list(mod.named_children()):
                _wrap_bn(child)
                if isinstance(child, nn.BatchNorm2d):
                    wrapped = BNWithGate(child, group_size=group_size, tau=tau,
                                         init_logit=init_logit, hard_eval=hard_eval)
                    setattr(mod, child_name, wrapped)
                # Downsample path: keep as is; its BN will also be wrapped by recursion
        _wrap_bn(m)
        return m

    # ---- logits getter ----
    @staticmethod
    def get_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return model(x)

    # ---- export (keep-all) ----
    @staticmethod
    @torch.no_grad()
    def export_keepall(model_with_gates: nn.Module) -> nn.Module:
        slim = deepcopy_eval_cpu(model_with_gates)

        def _unwrap(mod: nn.Module):
            for name, child in list(mod.named_children()):
                _unwrap(child)
                if isinstance(child, BNWithGate):
                    bn = child.bn
                    setattr(mod, name, copy.deepcopy(bn))
        _unwrap(slim)
        return slim

    # ---- export (pruned) ----
    @staticmethod
    @torch.no_grad()
    def export_pruned(model_with_gates: nn.Module, policy: ResNetExportPolicy | CoreExportPolicy, step: int) -> nn.Module:
        # Accept CoreExportPolicy as well (use same rounding for all BNs)
        if isinstance(policy, CoreExportPolicy):
            pol = ResNetExportPolicy(warmup_steps=policy.warmup_steps, rounding=policy.rounding,
                                     min_keep_ratio=getattr(policy.rounding, "min_keep_ratio", 0.0))
        else:
            pol = policy

        slim = deepcopy_eval_cpu(model_with_gates)

        # 1) Stem
        bn1 = getattr(slim, "bn1", None)
        if isinstance(bn1, BNWithGate):
            # grp = _choose_group_indices(bn1, pol, step)
            # ch_idx = torch.cat([torch.arange(i * bn1.group_size, (i + 1) * bn1.group_size) for i in grp]).long()
            ch_idx = _choose_group_indices(bn1, pol, tag="stem.bn1")
            
            slim.conv1 = _slice_conv2d(slim.conv1, keep_out=ch_idx)
            slim.bn1 = _slice_bn2d(bn1.bn, ch_idx)
            in_idx = ch_idx
        else:
            # No gates found → keep-all channels
            in_idx = torch.arange(slim.conv1.out_channels)

        # Helper to process one residual block (BasicBlock or Bottleneck)
        def _prune_block(blk, in_idx, *, force_groups=None, stage_tag="", bidx=0):
            gs = blk.bn1.group_size  # same for bn2 by construction
        
            if blk.downsample is not None:
                # first block in stage: free to change width
                ch1 = _choose_group_indices(blk.bn1, policy, tag=f"{stage_tag}.blk{bidx}.bn1")
                ch2 = _choose_group_indices(blk.bn2, policy, tag=f"{stage_tag}.blk{bidx}.bn2")
            else:
                # non-downsample block: MUST preserve identity width
                g_in = len(in_idx) // gs
                ch1 = _choose_group_indices(blk.bn1, policy, force_groups=g_in, tag=f"{stage_tag}.blk{bidx}.bn1")
                ch2 = _choose_group_indices(blk.bn2, policy, force_groups=g_in, tag=f"{stage_tag}.blk{bidx}.bn2")
        
            # conv1/bn1
            blk.conv1 = _slice_conv2d(blk.conv1, keep_in=in_idx, keep_out=ch1)
            blk.bn1   = _slice_bn2d(blk.bn1.bn, ch1)
        
            # conv2/bn2
            blk.conv2 = _slice_conv2d(blk.conv2, keep_in=ch1, keep_out=ch2)
            blk.bn2   = _slice_bn2d(blk.bn2.bn, ch2)
        
            # residual
            if blk.downsample is not None:
                ds_conv, ds_gbn = blk.downsample[0], blk.downsample[1]
                blk.downsample = nn.Sequential(
                    _slice_conv2d(ds_conv, keep_in=in_idx, keep_out=ch2),
                    _slice_bn2d(ds_gbn.bn, ch2),
                )
            else:
                # identity path -> output width must match input width
                assert ch2.numel() == in_idx.numel(), "Residual width mismatch without downsample."
        
            return blk, ch2


        # 2) Stages
        for lname in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(slim, lname, None)
            if layer is None:
                continue
                
            new_blocks: List[nn.Module] = []
            cur_in = in_idx
            force_groups: Optional[int] = None
            for bi, blk in enumerate(layer):
                has_ds = getattr(blk, "downsample", None) is not None
                gs = blk.bn1.group_size
                if has_ds:
                    # first block in stage may change width → choose freely & record groups for the stage
                    blk, out_idx = _prune_block(blk, cur_in, stage_tag=lname, bidx=bi)
                    force_groups = int(out_idx.numel() // (blk.bn2.group_size if isinstance(blk.bn2, BNWithGate) else 1))
                    cur_in = out_idx
                else:
                    blk, out_idx = _prune_block(blk, cur_in, force_groups=len(cur_in)//gs, stage_tag=lname, bidx=bi)
                    cur_in = out_idx
                new_blocks.append(blk)
            setattr(slim, lname, nn.Sequential(*new_blocks))
            in_idx = cur_in

        # 3) FC
        if hasattr(slim, "fc") and slim.fc.in_features != in_idx.numel():
            keep = torch.arange(in_idx.numel())
            W = slim.fc.weight.data[:, keep]
            new_fc = nn.Linear(in_idx.numel(), slim.fc.out_features, bias=(slim.fc.bias is not None))
            new_fc.weight.data.copy_(W)
            if slim.fc.bias is not None:
                new_fc.bias.data.copy_(slim.fc.bias.data)
            slim.fc = new_fc

        return slim

    # ---- BN recalibration ----
    @staticmethod
    @torch.no_grad()
    def bn_recalibration(model: nn.Module, loader, num_batches: int = 200, device: str = "cuda") -> None:
        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
        it = 0
        for batch in loader:
            if it >= num_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            _ = model(x.to(device, non_blocking=True))
            it += 1
        model.eval()

    # ---- optimizer param groups ----
    @staticmethod
    def param_groups(model: nn.Module):
        gates, bn_affine, conv_w, conv_b, fc = [], [], [], [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.endswith(".gate.logits"):
                gates.append(p)
            elif ".bn.weight" in n or ".bn.bias" in n:
                bn_affine.append(p)
            elif ".conv" in n and n.endswith(".weight"):
                conv_w.append(p)
            elif ".conv" in n and n.endswith(".bias"):
                conv_b.append(p)
            elif n.startswith("fc."):
                fc.append(p)
        return [
            {"params": gates,     "lr": 5e-3, "weight_decay": 0.0},
            {"params": conv_w,    "lr": 1e-4, "weight_decay": 1e-4},
            {"params": conv_b,    "lr": 1e-4, "weight_decay": 0.0},
            {"params": bn_affine, "lr": 3e-4, "weight_decay": 0.0},
            {"params": fc,        "lr": 5e-4, "weight_decay": 1e-4},
        ]




# ------------------------------ ResNet Proxy ------------------------------

@dataclass
class ResNetProxyConfig:
    scale_ms: float = 1.0
    alpha_conv: float = 1.0   # weight for conv FLOPs term


class ResNetLatencyProxy(LatencyProxy):
    """Latency proxy for ResNet-like backbones with BN gates.

    Approximates latency with a FLOPs-style sum over convs, using the *expected*
    kept channels after each BN gate (probs.sum()*group_size). Falls back to the
    full channel count when a gate is not found.

    Accepts a batch or an explicit (N,C,H,W) shape.
    """

    def __init__(self, cfg: Optional[ResNetProxyConfig] = None):
        super().__init__()
        self.cfg = cfg or ResNetProxyConfig()

    @staticmethod
    def _as_const_like_resnet(x_like: torch.Tensor, val):
        return torch.as_tensor(val, device=x_like.device, dtype=x_like.dtype)

    @staticmethod
    def _find_anchor_param(model: nn.Module) -> torch.Tensor:
        # Prefer any gate-like parameter; otherwise any parameter; else cpu scalar
        for m in model.modules():
            for nm in ("logits", "head_gate"):
                t = getattr(m, nm, None)
                if isinstance(t, torch.Tensor):
                    return t
        for p in model.parameters():
            return p
        return torch.tensor(0.0)

    @staticmethod
    def _kept_from_gate(module, anchor: torch.Tensor) -> Optional[torch.Tensor]:
        """Return expected kept channels for a BN gate: probs.sum() * group_size.
        If no gate is found, return None.
        """
        g = None
        for nm in ("gate", "neuron_gate", "channel_gate", "bn_gate"):
            if hasattr(module, nm):
                g = getattr(module, nm)
                break
        if g is None and hasattr(module, "logits") and hasattr(module, "tau"):
            g = module

        if g is None or not hasattr(g, "logits"):
            return None
        logits = g.logits
        tau = float(getattr(g, "tau", 1.5))
        group = int(getattr(g, "group", getattr(g, "group_size", 1)))
        if group <= 0: group = 1
        probs = torch.sigmoid(logits / tau)
        return probs.sum() * _as_const_like_resnet(anchor, group)

    def _add_cost(self, cost_like: torch.Tensor, oc, ic, k, stride, H, W):
        alpha = _as_const_like_resnet(cost_like, self.cfg.alpha_conv)
        # update spatial dims with conv stride (roughly, ignoring padding effects)
        H = (H + stride - 1) // stride
        W = (W + stride - 1) // stride
        flops = _as_const_like_resnet(cost_like, oc) * _as_const_like_resnet(cost_like, ic) * (k * k) * _as_const_like_resnet(cost_like, H) * _as_const_like_resnet(cost_like, W)
        return cost_like + alpha * flops, H, W

    def _predict_raw(self, model: nn.Module, sample: TensorOrBatch, **_) -> torch.Tensor:
        N, C_in, H0, W0 = _nchw_from_batch(sample)
        anchor = _find_anchor_param(model)
        cost = _as_const_like_resnet(anchor, 0.0)
        H = _as_const_like_resnet(anchor, int(H0))
        W = _as_const_like_resnet(anchor, int(W0))

        # Stem
        conv1 = getattr(model, "conv1")
        bn1 = getattr(model, "bn1", None)
        k = conv1.kernel_size[0]
        s = conv1.stride[0]
        kept_out = None
        if bn1 is not None:
            kept = _kept_from_gate(bn1, anchor)
            if kept is not None:
                kept_out = kept
        oc_eff = kept_out if kept_out is not None else _as_const_like_resnet(anchor, conv1.out_channels)
        cost, H, W = self._add_cost(cost, oc_eff, _as_const_like_resnet(anchor, C_in), k, s, H, W)
        in_ch = oc_eff

        def _block_cost(block, in_ch, H, W, cost):
            # conv1 -> bn1
            c1 = block.conv1
            b1 = block.bn1 if hasattr(block, "bn1") else None
            k1, s1 = c1.kernel_size[0], c1.stride[0]
            oc1_eff = _kept_from_gate(b1, anchor) or _as_const_like_resnet(anchor, c1.out_channels)
            cost, H, W = self._add_cost(cost, oc1_eff, in_ch, k1, s1, H, W)

            # conv2 -> bn2
            c2 = block.conv2
            b2 = block.bn2 if hasattr(block, "bn2") else None
            k2, s2 = c2.kernel_size[0], c2.stride[0]
            oc2_eff = _kept_from_gate(b2, anchor) or _as_const_like_resnet(anchor, c2.out_channels)
            cost, H, W = self._add_cost(cost, oc2_eff, oc1_eff, k2, s2, H, W)

            return oc2_eff, H, W, cost

        # Layers
        for lname in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(model, lname, None)
            if layer is None:
                continue
            for blk in layer:
                in_ch, H, W, cost = _block_cost(blk, in_ch, H, W, cost)

        scale = _as_const_like_resnet(anchor, self.cfg.scale_ms)
        return cost * scale

    @torch.no_grad()
    def calibrate(self, model: nn.Module, keepall_export_fn, profiler_fn, sample: TensorOrBatch, device: str = "cuda") -> float:
        """Calibrate `scale_ms` so proxy(model_keepall) ~= real latency in ms."""
        keep = keepall_export_fn(model)
        sample_shape = _nchw_from_batch(sample)
        mean_ms, _ = profiler_fn(keep, sample_shape, device=device)
        soft = float(self.predict(model, sample).detach().cpu())
        self.cfg.scale_ms = mean_ms / max(soft, 1e-9)
        return mean_ms

