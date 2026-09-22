# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental SM12xNvfp4Runner adapter for FlashInfer 263c4812027f.

This is separate from B12xMoEWrapper. It uses the unified MoELayer API with
exactly one candidate: Q0/route -> FC1/activation/Q1 -> FC2/finalize.
Only winner selection is specialized: the single fixed tactic must not enter
FlashInfer's world-group benchmark while pipeline stages wait for activations.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)
_clamp_warning_logged = False


def validate_unified_nvfp4_config(config: MoeRunnerConfig) -> None:
    """Validate the experimental BF16-in/out, pre-routed execution contract."""
    if config.activation != "silu" or not config.is_gated:
        raise ValueError("SM12x unified NVFP4 currently supports only gated SiLU.")
    if config.apply_router_weight_on_input or config.no_combine:
        raise ValueError(
            "SM12x unified NVFP4 requires output-side routing and combine."
        )
    if config.num_fused_shared_experts or config.params_dtype != torch.bfloat16:
        raise ValueError(
            "SM12x unified NVFP4 requires BF16 and no fused shared experts."
        )
    if config.gemm1_alpha is not None or config.gemm1_beta is not None:
        raise ValueError("SM12x unified NVFP4 requires default SwiGLU parameters.")
    global _clamp_warning_logged
    if (
        config.swiglu_limit is not None or config.gemm1_clamp_limit is not None
    ) and not _clamp_warning_logged:
        logger.warning(
            "EXPERIMENTAL SM12xNvfp4Runner: SwiGLU clamp is NOT applied, as in "
            "the previous NVFP4 comparison. This is not full DSv4 clamp semantics."
        )
        _clamp_warning_logged = True


def resolve_unified_nvfp4_scales(
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w13_input_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Map decode scales to the unified runner without requantizing weights.

    W13 global scales are [E,2] in checkpoint [gate,up] order, or [E].
    Q0/Q1 take scalar *quantization multipliers* (reciprocal decode scales).
    FC1 accepts independent per-expert gate/up alphas. FC2 only accepts a
    scalar alpha, so factor its per-expert weight scale as max_scale * ratio[e]
    and apply ratio[e] to the existing output-side routing weight.
    """
    experts = w2_weight_scale.numel()
    if w13_weight_scale.shape == (experts,):
        gate = up = w13_weight_scale
    elif w13_weight_scale.shape in ((experts, 1), (experts, 2)):
        gate = w13_weight_scale[:, 0]
        up = w13_weight_scale[:, -1]
    else:
        raise ValueError("Expected W13 global scales [E], [E,1], or [E,2].")
    if w2_weight_scale.shape != (experts,):
        raise ValueError("Expected W2 global scales [E].")
    for scale in (w13_weight_scale, w2_weight_scale, w13_input_scale, w2_input_scale):
        if scale.numel() == 0 or not torch.all(torch.isfinite(scale) & (scale > 0)):
            raise ValueError("NVFP4 global scales must be finite and positive.")
    with torch.inference_mode(False):
        a0 = w13_input_scale.float().max().reshape(1)
        a1 = w2_input_scale.float().max().reshape(1)
        down = w2_weight_scale.float()
        down_max = down.max().reshape(1)
        return {
            "w1_up_scale": (up.float() * a0).contiguous(),
            "w1_gate_scale": (gate.float() * a0).contiguous(),
            "w2_alpha": (down_max * a1).contiguous(),
            "q0_global_scale": a0.reciprocal().contiguous(),
            "q1_global_scale": a1.reciprocal().contiguous(),
            "route_scale": (down / down_max).contiguous(),
        }


def prepare_unified_nvfp4_weights(layer: torch.nn.Module) -> None:
    """Keep packed [up,gate] FP4 weights; prepare 128x4-swizzled E4M3 bytes."""
    from flashinfer.fused_moe import MoEWeightPack, SM12xNvfp4Config

    from sglang.srt.layers.quantization.utils import swizzle_blockscale
    from sglang.srt.layers.utils import copy_or_rebind_param

    if layer._sm12x_nvfp4_layer is not None:
        raise NotImplementedError("Restart the server to reload SM12x NVFP4 weights.")
    if layer.moe_ep_size != 1 or layer.moe_tp_size != 1:
        raise ValueError("SM12x unified NVFP4 currently requires TP1/EP1.")
    if (
        not layer.quant_config.is_checkpoint_nvfp4_serialized
        or layer.quant_config.use_per_token_activation
    ):
        raise ValueError("SM12x unified NVFP4 requires serialized static-scale NVFP4.")
    if layer.hidden_size % 128 or layer.intermediate_size_per_partition % 128:
        raise ValueError("SM12x unified NVFP4 requires H and I aligned to 128.")

    scales = resolve_unified_nvfp4_scales(
        layer.w13_weight_scale_2,
        layer.w2_weight_scale_2,
        layer.w13_input_scale,
        layer.w2_input_scale,
    )
    for name, scale in scales.items():
        copy_or_rebind_param(layer, f"sm12x_unified_{name}", scale)
    for prefix in ("w13", "w2"):
        weight = getattr(layer, f"{prefix}_weight")
        scale = getattr(layer, f"{prefix}_weight_scale")
        if weight.dtype != torch.uint8 or scale.dtype != torch.float8_e4m3fn:
            raise ValueError("Expected packed uint8 weights and E4M3 block scales.")
        # Unlike B12x, this runner consumes the swizzled 128x4 layout directly,
        # not convert_sf_to_mma_layout(). Expert rows are aligned to 128.
        swizzled = (
            swizzle_blockscale(scale)
            .view(torch.uint8)
            .reshape(weight.shape[0] * weight.shape[1], weight.shape[2] * 2 // 16)
        )
        copy_or_rebind_param(layer, f"{prefix}_blockscale_swizzled", swizzled)
        setattr(
            layer,
            f"{prefix}_weight_scale",
            getattr(layer, f"{prefix}_blockscale_swizzled"),
        )

    native = SM12xNvfp4Config.prepare_weights(
        w1_weight=layer.w13_weight,
        w1_weight_sf=layer.w13_blockscale_swizzled,
        w1_up_scale=layer.sm12x_unified_w1_up_scale,
        w1_gate_scale=layer.sm12x_unified_w1_gate_scale,
        w2_weight=layer.w2_weight,
        w2_weight_sf=layer.w2_blockscale_swizzled,
        w2_alpha=layer.sm12x_unified_w2_alpha,
        q0_global_scale=layer.sm12x_unified_q0_global_scale,
        q1_global_scale=layer.sm12x_unified_q1_global_scale,
    )
    layer._sm12x_nvfp4_weights = MoEWeightPack()
    layer._sm12x_nvfp4_weights.prepare_for("sm12x_nvfp4", native)
    layer.dispatcher.set_quant_config({"input_global_scale": None})


def _select_single_nvfp4_runner(act_pack, weight_pack, runners):
    """Select the pinned runner's sole tactic without cross-rank timing.

    FlashInfer benchmarks even a single candidate, then gathers timing
    over the default distributed world. In PP, downstream stages are waiting for
    our activations and cannot join that collective. Keep pack_inputs validation
    and fail closed if a future runner introduces a choice requiring real tuning.
    """
    from flashinfer.fused_moe import SM12xNvfp4Runner

    if len(runners) != 1 or not isinstance(runners[0], SM12xNvfp4Runner):
        raise RuntimeError("PP-safe selection requires exactly SM12xNvfp4Runner.")
    runner = runners[0]
    inputs = runner.pack_inputs(act_pack, weight_pack)
    if runner.get_valid_tactics(inputs, profile=None) != [-1]:
        raise RuntimeError("PP-safe SM12x selection requires the sole tactic -1.")
    return runner, -1


def unified_nvfp4_validate_expert_ids(layer: torch.nn.Module) -> bool:
    """Allow trusted IDs only for the audited DSv4 Flash routing configuration.

    The opt-out assumes the loaded HashTopK tables have been range-validated.
    TP1/EP1 with trivial placement does not remap expert IDs or mask padded
    routing rows to -1. This is not a general promise for other MoE backends.
    """
    if envs.SGLANG_SM12X_NVFP4_VALIDATE_EXPERT_IDS.get():
        return True
    from sglang.srt.runtime_context import (
        get_exec,
        get_parallel,
        process_model_config,
    )

    hf_config = process_model_config().hf_config
    parallel = get_parallel()
    moe = get_exec().moe
    if not (
        hf_config.model_type == "deepseek_v4"
        and layer.num_experts == layer.num_local_experts == 256
        and layer.moe_runner_config.top_k == 6
        and layer.hidden_size == 4096
        and layer.intermediate_size_per_partition == 2048
        and layer.moe_tp_size == layer.moe_ep_size == 1
        and parallel.tp_size == parallel.dp_size == 1
        and not moe.enable_eplb
        and moe.init_expert_location == "trivial"
        and moe.ep_num_redundant_experts == 0
        and not moe.enable_waterfill
        and layer.moe_runner_config.num_fused_shared_experts == 0
    ):
        raise ValueError(
            "Disabling SM12x expert-ID validation requires the verified DSv4 Flash "
            "TP1/EP1/DP1 configuration with no EPLB, redundant or fused shared experts."
        )
    return False


def configure_unified_nvfp4_validation(layer: torch.nn.Module) -> bool:
    """Map the audited SGLang policy to the upstream process-level switch.

    Set this before constructing any SM12x runners: FlashInfer snapshots the
    environment in its constructor. Refuse conflicting explicit settings rather
    than silently disabling validation or changing another layer's policy.
    """
    validate_expert_ids = unified_nvfp4_validate_expert_ids(layer)
    name = "FLASHINFER_SM12X_NVFP4_VALIDATE_TOPK_IDS"
    existing = os.environ.get(name)
    if existing is not None and (existing not in ("0", "")) != validate_expert_ids:
        raise ValueError(
            f"{name} conflicts with SGLANG_SM12X_NVFP4_VALIDATE_EXPERT_IDS; "
            "set both to the same 0/1 value or leave the FlashInfer variable unset."
        )
    os.environ[name] = "1" if validate_expert_ids else "0"
    return validate_expert_ids


def ensure_unified_nvfp4_layer(layer: torch.nn.Module) -> None:
    """Build the public MoELayer with only SM12xNvfp4Runner eligible."""
    if layer._sm12x_nvfp4_layer is not None:
        return
    from flashinfer.fused_moe import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        MoELayer,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
        SM12xNvfp4Config,
        SM12xNvfp4Runner,
        SwiGLU,
    )

    from sglang.srt.runtime_context import cutedsl_moe_max_num_tokens

    validate_expert_ids = configure_unified_nvfp4_validation(layer)
    config = MoEConfig(
        routing=RoutingConfig(
            num_experts=layer.num_experts, top_k=layer.moe_runner_config.top_k
        ),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        experts=ExpertConfig(
            intermediate_size=layer.intermediate_size_per_partition,
            local_num_experts=layer.num_local_experts,
        ),
        activation=SwiGLU(),
        backend=BackendOptions(candidates=(SM12xNvfp4Config(),)),
        execution=ExecutionConfig(
            enable_pdl=False, tune_max_num_tokens=cutedsl_moe_max_num_tokens()
        ),
    )

    class PPSafeSingleRunnerMoELayer(MoELayer):
        # Preserve MoELayer.__call__: token ceiling, routing-mode checks,
        # per-bucket cache, pack_inputs validation and runner.forward launches.
        # Do not monkey-patch global FlashInfer timing/distributed functions.
        def _select_winner(self, act_pack, weight_pack, runners):
            return _select_single_nvfp4_runner(act_pack, weight_pack, runners)

    moe = PPSafeSingleRunnerMoELayer(config, device=layer.w13_weight.device)
    if len(moe.runners) != 1 or not isinstance(moe.runners[0], SM12xNvfp4Runner):
        raise RuntimeError("Expected exactly SM12xNvfp4Runner; refusing a fallback.")
    if getattr(moe.runners[0], "_validate_topk_ids", None) is not validate_expert_ids:
        raise RuntimeError(
            "SM12xNvfp4Runner did not honor the upstream validation switch; "
            "install the pinned FlashInfer 263c4812 revision."
        )
    layer._sm12x_nvfp4_layer = moe
    logger.info(
        "SM12x unified NVFP4 enabled: runner=SM12xNvfp4Runner, E=%s K=%s H=%s I=%s, "
        "selection=single-tactic-no-cross-rank-timing, validate_expert_ids=%s, "
        "enable_pdl=False",
        layer.num_experts,
        layer.moe_runner_config.top_k,
        layer.hidden_size,
        layer.intermediate_size_per_partition,
        validate_expert_ids,
    )


@dataclass
class UnifiedNvfp4MoeQuantInfo(MoeQuantInfo):
    layer: torch.nn.Module


@register_fused_func("none", "flashinfer_sm12x_nvfp4")
def fused_experts_unified_nvfp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: UnifiedNvfp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Run BF16 [T,H] with existing SGLang [T,K] routing and FP4 weights."""
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    x = dispatch_output.hidden_states
    if x.dtype != torch.bfloat16 or dispatch_output.hidden_states_scale is not None:
        raise ValueError("SM12x unified NVFP4 expects unquantized BF16 activations.")
    if x.shape[0] == 0:
        return StandardCombineInput(hidden_states=x)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SM12xNvfp4Runner currently requires --disable-cuda-graph.")
    topk = dispatch_output.topk_output
    if not TopKOutputChecker.format_is_standard(topk):
        raise ValueError("SM12x unified NVFP4 expects standard precomputed top-k.")
    from flashinfer.fused_moe import MoEActivationPack

    layer = quant_info.layer
    ensure_unified_nvfp4_layer(layer)
    ids = topk.topk_ids.to(torch.int32).contiguous()
    # Upstream treats -1 as an inactive route. Tensor indexing maps it to the
    # last finite expert scale, but that weight is never consumed by routing.
    # Keep real expert IDs unchanged; do not clamp malformed IDs into validity.
    weights = topk.topk_weights.float() * layer.sm12x_unified_route_scale[ids.long()]
    act = MoEActivationPack(x, None, ids, weights.contiguous())
    output = layer._sm12x_nvfp4_layer(act, layer._sm12x_nvfp4_weights)
    return StandardCombineInput(hidden_states=output)
