from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rotationquant.modeling import iter_llama_decoder_layers
from rotationquant.quantizers import gaussian_lloyd_max_codebook, symmetric_absmax_quantize
from rotationquant.rotations import fwht


@dataclass(frozen=True)
class StageBMethod:
    name: str
    rotation: str
    quantizer: str
    compute_interpretation: str


@dataclass(frozen=True)
class WABitSpec:
    method: str
    w_bits: int
    a_bits: int

    @property
    def label(self) -> str:
        return f"W{self.w_bits}A{self.a_bits}"


STAGE_B_METHODS: dict[str, StageBMethod] = {
    "direct_absmax": StageBMethod(
        name="direct_absmax",
        rotation="none",
        quantizer="absmax",
        compute_interpretation="uniform fake quant; future INT mapping is plausible",
    ),
    "rot_absmax": StageBMethod(
        name="rot_absmax",
        rotation="block_hadamard_last_dim",
        quantizer="absmax",
        compute_interpretation="rotation + uniform fake quant; needs rotation handling",
    ),
    "rot_lm": StageBMethod(
        name="rot_lm",
        rotation="block_hadamard_last_dim",
        quantizer="gaussian_lloyd_max",
        compute_interpretation="non-uniform codebook fake quant; not native INT GEMM",
    ),
}

STAGE_B_LINEAR_SPECS: dict[str, WABitSpec] = {
    "direct_absmax_w4a4": WABitSpec("direct_absmax", w_bits=4, a_bits=4),
    "rot_absmax_w4a4": WABitSpec("rot_absmax", w_bits=4, a_bits=4),
    "rot_lm_w4a4": WABitSpec("rot_lm", w_bits=4, a_bits=4),
    "rot_lm_w3a4": WABitSpec("rot_lm", w_bits=3, a_bits=4),
    "rot_lm_w4a3": WABitSpec("rot_lm", w_bits=4, a_bits=3),
    "rot_lm_w3a3": WABitSpec("rot_lm", w_bits=3, a_bits=3),
    "rot_lm_w2a4": WABitSpec("rot_lm", w_bits=2, a_bits=4),
}

STAGE_B_FFN_SPECS: dict[str, WABitSpec] = {
    "ffn_direct_absmax_w4a4": WABitSpec("direct_absmax", w_bits=4, a_bits=4),
    "ffn_rot_absmax_w4a4": WABitSpec("rot_absmax", w_bits=4, a_bits=4),
    "ffn_rot_lm_w4a4": WABitSpec("rot_lm", w_bits=4, a_bits=4),
    "ffn_rot_lm_w3a4": WABitSpec("rot_lm", w_bits=3, a_bits=4),
    "ffn_rot_lm_w4a3": WABitSpec("rot_lm", w_bits=4, a_bits=3),
    "ffn_rot_lm_w3a3": WABitSpec("rot_lm", w_bits=3, a_bits=3),
}

STAGE_B_MODEL_METHODS: dict[str, WABitSpec] = {
    "ffn_direct_absmax_w4a4": STAGE_B_FFN_SPECS["ffn_direct_absmax_w4a4"],
    "ffn_rot_absmax_w4a4": STAGE_B_FFN_SPECS["ffn_rot_absmax_w4a4"],
    "ffn_rot_lm_w4a4": STAGE_B_FFN_SPECS["ffn_rot_lm_w4a4"],
    "ffn_rot_lm_w3a4": STAGE_B_FFN_SPECS["ffn_rot_lm_w3a4"],
    "ffn_rot_lm_w4a3": STAGE_B_FFN_SPECS["ffn_rot_lm_w4a3"],
}


def _pad_last_dim(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    pad = (-x.shape[-1]) % block_size
    if pad:
        x = F.pad(x, (0, pad))
    return x, pad


def block_hadamard_last_dim(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Apply orthonormal Hadamard independently to blocks along the last dim."""
    padded, pad = _pad_last_dim(x, block_size)
    leading_shape = padded.shape[:-1]
    blocks = padded.reshape(*leading_shape, padded.shape[-1] // block_size, block_size)
    rotated = fwht(blocks, dim=-1, normalize=True).reshape(*leading_shape, padded.shape[-1])
    if pad:
        rotated = rotated[..., :-pad]
    return rotated


def _block_rms_lloyd_max_quantize(x: torch.Tensor, bits: int, block_size: int) -> torch.Tensor:
    padded, pad = _pad_last_dim(x.float(), block_size)
    leading_shape = padded.shape[:-1]
    blocks = padded.reshape(*leading_shape, padded.shape[-1] // block_size, block_size)
    rms = blocks.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
    normalized = blocks / rms

    codebook = torch.tensor(gaussian_lloyd_max_codebook(bits), device=x.device, dtype=torch.float32)
    boundaries = (codebook[:-1] + codebook[1:]) / 2
    try:
        indices = torch.bucketize(normalized, boundaries)
    except (RuntimeError, NotImplementedError):
        # Some PyTorch/MPS combinations lack bucketize coverage. This fallback
        # keeps model-level fake quant numerically available, albeit slower.
        indices = torch.bucketize(normalized.cpu(), boundaries.cpu()).to(device=x.device)
    quantized = codebook[indices].reshape_as(blocks) * rms
    restored = quantized.reshape(*leading_shape, padded.shape[-1])
    if pad:
        restored = restored[..., :-pad]
    return restored.to(dtype=x.dtype)


def quantize_stage_b_domain(
    x: torch.Tensor,
    bits: int,
    quantizer: str,
    block_size: int = 128,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Fake-quantize a tensor already in the domain selected by the method."""
    if quantizer == "absmax":
        result = symmetric_absmax_quantize(x, bits)
        return result.values.to(dtype=x.dtype), {
            "quantizer_type": result.quantizer_type,
            "scale_granularity": "tensor",
            **result.metadata,
        }
    if quantizer == "gaussian_lloyd_max":
        return _block_rms_lloyd_max_quantize(x, bits, block_size), {
            "quantizer_type": "non-uniform codebook",
            "scale_granularity": "block_rms",
            "levels": 1 << bits,
            "codebook": "gaussian_lloyd_max_standard_normal",
        }
    raise ValueError(f"Unsupported Stage B quantizer: {quantizer}")


def quantize_activation_for_b1(
    x: torch.Tensor,
    method_name: str,
    bits: int,
    block_size: int = 128,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return an activation reconstructed to the original domain for B1 metrics."""
    method = STAGE_B_METHODS[method_name]
    if method.rotation == "none":
        quantized, metadata = quantize_stage_b_domain(x, bits, method.quantizer, block_size)
        return quantized, {
            "method": method.name,
            "bits": bits,
            "activation_bits": bits,
            "rotation": method.rotation,
            "compute_interpretation": method.compute_interpretation,
            **metadata,
        }

    rotated = block_hadamard_last_dim(x, block_size=block_size)
    quantized_rotated, metadata = quantize_stage_b_domain(rotated, bits, method.quantizer, block_size)
    restored = block_hadamard_last_dim(quantized_rotated, block_size=block_size)
    return restored.to(dtype=x.dtype), {
        "method": method.name,
        "bits": bits,
        "activation_bits": bits,
        "rotation": method.rotation,
        "compute_interpretation": method.compute_interpretation,
        "block_size": block_size,
        **metadata,
    }


def _quantize_linear_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    spec: WABitSpec,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    method = STAGE_B_METHODS[spec.method]
    if method.rotation == "none":
        x_domain = x
        weight_domain = weight
    else:
        x_domain = block_hadamard_last_dim(x, block_size=block_size)
        weight_domain = block_hadamard_last_dim(weight, block_size=block_size)

    x_quant, x_meta = quantize_stage_b_domain(x_domain, spec.a_bits, method.quantizer, block_size)
    w_quant, w_meta = quantize_stage_b_domain(weight_domain, spec.w_bits, method.quantizer, block_size)
    metadata = {
        "method": method.name,
        "bits": spec.label,
        "w_bits": spec.w_bits,
        "a_bits": spec.a_bits,
        "rotation": method.rotation,
        "compute_interpretation": method.compute_interpretation,
        "activation_quantizer_type": x_meta["quantizer_type"],
        "weight_quantizer_type": w_meta["quantizer_type"],
        "activation_scale_granularity": x_meta["scale_granularity"],
        "weight_scale_granularity": w_meta["scale_granularity"],
    }
    return x_quant, w_quant, metadata


def prepare_stage_b_weight(
    weight: torch.Tensor,
    spec: WABitSpec,
    block_size: int = 128,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Pre-quantize a Linear weight in the method domain for model-level use."""
    method = STAGE_B_METHODS[spec.method]
    weight_domain = weight if method.rotation == "none" else block_hadamard_last_dim(weight, block_size=block_size)
    weight_quant, weight_meta = quantize_stage_b_domain(weight_domain, spec.w_bits, method.quantizer, block_size)
    return weight_quant.to(dtype=weight.dtype), {
        "method": method.name,
        "bits": spec.label,
        "w_bits": spec.w_bits,
        "rotation": method.rotation,
        "weight_quantizer_type": weight_meta["quantizer_type"],
        "weight_scale_granularity": weight_meta["scale_granularity"],
    }


def quantize_stage_b_activation_domain(
    x: torch.Tensor,
    spec: WABitSpec,
    block_size: int = 128,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Rotate when needed, then fake-quantize an activation in method domain."""
    method = STAGE_B_METHODS[spec.method]
    x_domain = x if method.rotation == "none" else block_hadamard_last_dim(x, block_size=block_size)
    x_quant, x_meta = quantize_stage_b_domain(x_domain, spec.a_bits, method.quantizer, block_size)
    return x_quant.to(dtype=x.dtype), {
        "activation_quantizer_type": x_meta["quantizer_type"],
        "activation_scale_granularity": x_meta["scale_granularity"],
    }


def fake_quant_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    spec: WABitSpec,
    block_size: int = 128,
) -> tuple[torch.Tensor, dict[str, object]]:
    x_quant, w_quant, metadata = _quantize_linear_inputs(x, weight, spec, block_size)
    y = F.linear(x_quant, w_quant, bias)
    return y.to(dtype=x.dtype), metadata


def _apply_ffn_activation(act_owner: torch.nn.Module | None, gate: torch.Tensor) -> torch.Tensor:
    act_fn = getattr(act_owner, "act_fn", None) if act_owner is not None else None
    if act_fn is not None:
        return act_fn(gate)
    return F.silu(gate)


def fake_quant_ffn_from_weights(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    spec: WABitSpec,
    block_size: int = 128,
    gate_bias: torch.Tensor | None = None,
    up_bias: torch.Tensor | None = None,
    down_bias: torch.Tensor | None = None,
    act_owner: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    gate, gate_meta = fake_quant_linear(
        x,
        gate_weight,
        gate_bias,
        spec,
        block_size=block_size,
    )
    up, _ = fake_quant_linear(
        x,
        up_weight,
        up_bias,
        spec,
        block_size=block_size,
    )
    intermediate = _apply_ffn_activation(act_owner, gate) * up
    down, down_meta = fake_quant_linear(
        intermediate,
        down_weight,
        down_bias,
        spec,
        block_size=block_size,
    )
    metadata = {
        **gate_meta,
        "down_activation_quantizer_type": down_meta["activation_quantizer_type"],
        "down_weight_quantizer_type": down_meta["weight_quantizer_type"],
    }
    return down.to(dtype=x.dtype), metadata


def fake_quant_ffn(
    x: torch.Tensor,
    ffn_module: torch.nn.Module,
    spec: WABitSpec,
    block_size: int = 128,
) -> tuple[torch.Tensor, dict[str, object]]:
    return fake_quant_ffn_from_weights(
        x,
        ffn_module.gate_proj.weight,
        ffn_module.up_proj.weight,
        ffn_module.down_proj.weight,
        spec,
        block_size=block_size,
        gate_bias=getattr(ffn_module.gate_proj, "bias", None),
        up_bias=getattr(ffn_module.up_proj, "bias", None),
        down_bias=getattr(ffn_module.down_proj, "bias", None),
        act_owner=ffn_module,
    )


class StageBFFNWrapper(torch.nn.Module):
    """Model-level FFN-only fake quant wrapper for Stage B PPL experiments."""

    def __init__(self, ffn_module: torch.nn.Module, spec: WABitSpec, block_size: int = 128) -> None:
        super().__init__()
        self.spec = spec
        self.block_size = block_size
        self.act_fn = getattr(ffn_module, "act_fn", torch.nn.SiLU())
        gate_weight, _ = prepare_stage_b_weight(ffn_module.gate_proj.weight.detach(), spec, block_size=block_size)
        up_weight, _ = prepare_stage_b_weight(ffn_module.up_proj.weight.detach(), spec, block_size=block_size)
        down_weight, _ = prepare_stage_b_weight(ffn_module.down_proj.weight.detach(), spec, block_size=block_size)
        self.register_buffer("gate_weight", gate_weight)
        self.register_buffer("up_weight", up_weight)
        self.register_buffer("down_weight", down_weight)
        self.register_buffer("gate_bias", getattr(ffn_module.gate_proj, "bias", None))
        self.register_buffer("up_bias", getattr(ffn_module.up_proj, "bias", None))
        self.register_buffer("down_bias", getattr(ffn_module.down_proj, "bias", None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_input, _ = quantize_stage_b_activation_domain(x, self.spec, block_size=self.block_size)
        gate = F.linear(gate_input, self.gate_weight, self.gate_bias)
        up = F.linear(gate_input, self.up_weight, self.up_bias)
        intermediate = self.act_fn(gate) * up
        down_input, _ = quantize_stage_b_activation_domain(intermediate, self.spec, block_size=self.block_size)
        return F.linear(down_input, self.down_weight, self.down_bias).to(dtype=x.dtype)


def apply_stage_b_ffn_fake_quant_(
    model: torch.nn.Module,
    method_name: str,
    block_size: int = 128,
) -> list[dict[str, object]]:
    """Replace only FFN modules with Stage B fake-quant wrappers."""
    if method_name not in STAGE_B_MODEL_METHODS:
        raise ValueError(f"Unknown Stage B model method: {method_name}")
    spec = STAGE_B_MODEL_METHODS[method_name]
    records: list[dict[str, object]] = []
    for layer_index, layer in iter_llama_decoder_layers(model):
        original_ffn = layer.mlp
        layer.mlp = StageBFFNWrapper(original_ffn, spec=spec, block_size=block_size)
        method = STAGE_B_METHODS[spec.method]
        records.append(
            {
                "layer": f"model.layers.{layer_index}.mlp",
                "method": method.name,
                "bits": spec.label,
                "w_bits": spec.w_bits,
                "a_bits": spec.a_bits,
                "rotation": method.rotation,
                "compute_interpretation": method.compute_interpretation,
            }
        )
    return records
