import torch
from safetensors import safe_open
from typing import Dict, Any, Tuple

class QuantizationError(Exception):
    """Custom exception for quantization errors."""
    pass

def _get_tensor(tensor_info: Dict[str, Any]) -> torch.Tensor:
    """Opens a safetensors file and extracts a tensor."""
    try:
        with safe_open(tensor_info["file_path"], framework="pt", device="cpu") as f:
            return f.get_tensor(tensor_info["name"])
    except Exception as e:
        raise QuantizationError(f"Error loading tensor: {e}") from e

def _validate_tensor_for_quantization(tensor: torch.Tensor):
    """Checks if a tensor is suitable for quantization."""
    if tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise QuantizationError(f"Quantization only supported for float tensors. Selected tensor is {tensor.dtype}.")

def _get_per_tensor_params(tensor: torch.Tensor, quant_type: str) -> Tuple[torch.Tensor, int, torch.dtype]:
    """Calculates scale, zero_point, and qdtype for per-tensor quantization."""
    if quant_type == "Symmetric":
        q_min, q_max = -128, 127
        abs_max = torch.max(torch.abs(tensor.min()), torch.abs(tensor.max()))
        if abs_max.item() == 0:
             raise QuantizationError("Cannot quantize tensor with all zero values.")
        scale = abs_max / q_max
        zero_point = 0
        qdtype = torch.qint8
    else:  # Asymmetric
        q_min, q_max = 0, 255
        t_min, t_max = tensor.min(), tensor.max()
        if t_min.item() == t_max.item():
            if t_min.item() == 0:
                raise QuantizationError("Cannot quantize tensor with all zero values.")
            # Handle constant value tensor
            scale = 1.0
            zero_point = int(q_min + (t_min.item() / scale))
            if not (q_min <= zero_point <= q_max):
                raise QuantizationError(f"Cannot represent constant value {t_min.item()} with quantization parameters.")
        else:
            scale = (t_max - t_min) / (q_max - q_min)
            if scale.item() == 0:
                raise QuantizationError("Cannot quantize tensor with zero range.")
            zero_point = q_min - torch.round(t_min / scale)
            zero_point = int(zero_point.clamp(q_min, q_max).item())
        qdtype = torch.quint8

    return torch.tensor(scale), zero_point, qdtype

def _quantize_per_tensor(tensor: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """Performs per-tensor quantization."""
    quant_type = config.get("Quantization Type")
    scale, zero_point, qdtype = _get_per_tensor_params(tensor, quant_type)
    return torch.quantize_per_tensor(tensor, scale.item(), zero_point, qdtype)

def _get_per_channel_params(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Calculates scales, zero_points, and channel axis for per-channel quantization."""
    if tensor.ndim <= 1:
        raise QuantizationError("Per-channel quantization requires at least 2 dimensions.")

    ch_axis = 0
    dims_to_reduce = list(range(tensor.ndim))
    dims_to_reduce.remove(ch_axis)
    
    abs_max = torch.amax(torch.abs(tensor), dim=dims_to_reduce)
    scales = abs_max / 127.0
    scales[scales == 0] = 1.0
    zero_points = torch.zeros(tensor.shape[ch_axis], dtype=torch.long)
    return scales, zero_points, ch_axis

def _quantize_per_channel(tensor: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """Performs per-channel quantization."""
    quant_type = config.get("Quantization Type")
    if quant_type == "Asymmetric":
        raise QuantizationError("Per-channel asymmetric quantization is not supported by PyTorch.")

    scales, zero_points, ch_axis = _get_per_channel_params(tensor)
    return torch.quantize_per_channel(tensor, scales, zero_points, ch_axis, torch.qint8)

def _quantize_per_group(tensor: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """Performs per-group quantization."""
    if config.get("Quantization Type") == "Asymmetric":
        raise QuantizationError("Per-group asymmetric quantization is not supported.")

    if tensor.ndim == 0:
        raise QuantizationError("Per-group quantization is not supported for 0-dim tensors.")

    group_size = config.get("Group Size")
    if not group_size or not isinstance(group_size, int) or group_size <= 0:
        raise QuantizationError(f"Group Size must be a positive integer, got {group_size}, type{type(group_size)}.")

    flat_tensor = tensor.flatten()
    num_elements = flat_tensor.numel()
    if num_elements % group_size != 0:
        raise QuantizationError(f"Number of elements ({num_elements}) must be divisible by Group Size ({group_size}).")

    grouped_tensor = flat_tensor.view(-1, group_size)

    scales = grouped_tensor.abs().max(dim=1, keepdim=True)[0] / 127
    scales[scales == 0] = 1.0
    zero_points = torch.zeros(scales.shape[0], dtype=torch.long)

    quantized_grouped = torch.quantize_per_channel(grouped_tensor, scales.flatten(), zero_points, 0, torch.qint8)

    return quantized_grouped

def _quantize_per_block(tensor: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """Performs per-block quantization."""
    if config.get("Quantization Type") == "Asymmetric":
        raise QuantizationError("Per-block asymmetric quantization is not supported.")

    if tensor.ndim != 2:
        raise QuantizationError("Per-block quantization is only supported for 2D tensors.")

    block_size = config.get("Block Size")
    if not block_size or not isinstance(block_size, int) or block_size <= 0:
        raise QuantizationError("Block Size must be a positive integer.")

    H, W = tensor.shape
    if H % block_size != 0 or W % block_size != 0:
        raise QuantizationError(f"Tensor dimensions ({H}, {W}) must be divisible by Block Size ({block_size}).")

    reshaped_T = tensor.view(H // block_size, block_size, W // block_size, block_size).transpose(1, 2).contiguous()
    reshaped_T = reshaped_T.view(-1, block_size * block_size)

    scales = reshaped_T.abs().max(dim=1, keepdim=True)[0] / 127
    scales[scales == 0] = 1.0
    zero_points = torch.zeros(scales.shape[0], dtype=torch.long)

    quantized_reshaped = torch.quantize_per_channel(reshaped_T, scales.flatten(), zero_points, 0, torch.qint8)

    return quantized_reshaped

def _calculate_metrics(original_tensor: torch.Tensor, quantized_tensor: torch.Tensor) -> Dict[str, Any]:
    """Calculates MSE and SNR for a quantized tensor."""
    dequantized_tensor = quantized_tensor.dequantize()
    mse = torch.mean((original_tensor - dequantized_tensor)**2)

    if mse > 0:
        snr = (10 * torch.log10(torch.mean(original_tensor**2) / mse)).item()
    else:
        snr = float('inf')

    return {"mse": mse.item(), "snr": snr}

def _format_results(
    quantized_tensor: torch.Tensor,
    original_dtype: torch.dtype,
    original_shape: str,
    original_min: float,
    original_max: float,
    metrics: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Formats the final quantization results dictionary."""
    scale_str = ""
    zp_str = ""
    if quantized_tensor.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
        scale_str = f"{quantized_tensor.q_scale():.6f}"
        zp_str = str(quantized_tensor.q_zero_point())
    else:
        scales = quantized_tensor.q_per_channel_scales()
        zero_points = quantized_tensor.q_per_channel_zero_points()
        scale_str = f"min: {scales.min().item():.6f}, max: {scales.max().item():.6f}"
        zp_str = f"min: {zero_points.min().item()}, max: {zero_points.max().item()}"

    algorithm_str = f"{config.get('Bit Width', '')} {config.get('Quantization Granularity')} {config.get('Quantization Type')}"

    return {
        "algorithm": algorithm_str,
        "original_dtype": str(original_dtype),
        "original_shape": original_shape,
        "original_min": original_min,
        "original_max": original_max,
        "quantized_dtype": str(quantized_tensor.dtype),
        "quantized_shape": ' × '.join(map(str, quantized_tensor.shape)),
        "scale": scale_str,
        "zero_point": zp_str,
        "mse": metrics["mse"],
        "snr": metrics["snr"],
    }

def quantize_tensor(tensor_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quantizes a tensor based on the provided configuration.

    Args:
        tensor_info: A dictionary containing information about the tensor,
                     including its file_path and name.
        config: A dictionary with the quantization settings.

    Returns:
        A dictionary with the quantization results and metrics.

    Raises:
        QuantizationError: If there is an error during quantization.
    """
    tensor = _get_tensor(tensor_info)

    original_dtype = tensor.dtype
    _validate_tensor_for_quantization(tensor)

    float_tensor = tensor.float()

    original_shape = ' × '.join(map(str, float_tensor.shape))
    original_min = float_tensor.min().item()
    original_max = float_tensor.max().item()

    granularity = config.get("Quantization Granularity")

    quantized_tensor = None
    if granularity == "Per-Tensor":
        quantized_tensor = _quantize_per_tensor(float_tensor, config)
    elif granularity == "Per-Channel":
        quantized_tensor = _quantize_per_channel(float_tensor, config)
    elif granularity == "Per-Group":
        quantized_tensor = _quantize_per_group(float_tensor, config)
    elif granularity == "Per-Block":
        quantized_tensor = _quantize_per_block(float_tensor, config)
    else:
        raise QuantizationError(f"{granularity} quantization is not yet implemented.")

    if quantized_tensor is None:
        raise QuantizationError("Quantization failed for an unknown reason.")

    if granularity in ["Per-Group", "Per-Block"]:
        dequantized_tensor = quantized_tensor.dequantize()
        if granularity == "Per-Group":
            dequantized_tensor = dequantized_tensor.view(float_tensor.shape)
        else:  # Per-Block
            block_size = config.get("Block Size")
            H, W = float_tensor.shape
            dequantized_view = dequantized_tensor.view(H // block_size, W // block_size, block_size, block_size)
            dequantized_transposed = dequantized_view.transpose(1, 2)
            dequantized_tensor = dequantized_transposed.contiguous().view(H, W)

        mse = torch.mean((float_tensor - dequantized_tensor)**2)
        if mse > 0:
            snr = (10 * torch.log10(torch.mean(float_tensor**2) / mse)).item()
        else:
            snr = float('inf')
        metrics = {"mse": mse.item(), "snr": snr}
    else:
        metrics = _calculate_metrics(float_tensor, quantized_tensor)

    return _format_results(
        quantized_tensor,
        original_dtype,
        original_shape,
        original_min,
        original_max,
        metrics,
        config
    )
