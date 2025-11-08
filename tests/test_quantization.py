import pytest
import torch
from pathlib import Path
from safetensors.torch import save_file
from typing import Dict, Any

from safe_view.quantization import quantize_tensor, QuantizationError

@pytest.fixture
def create_test_tensor_file(tmp_path: Path) -> Dict[str, Any]:
    """
    Creates a temporary safetensors file with a sample tensor
    and returns the tensor_info dictionary for it.
    """
    tensors = {
        "test_tensor": torch.randn(10, 20),
        "block_tensor": torch.randn(20, 20),
        "int_tensor": torch.randint(0, 100, (5, 5)),
        "zero_tensor": torch.zeros(4, 4),
        "1d_tensor": torch.randn(100),
    }
    file_path = tmp_path / "model.safetensors"
    save_file(tensors, file_path)
    
    tensor_info = {
        "name": "test_tensor",
        "file_path": str(file_path),
    }
    return tensor_info

def test_per_tensor_symmetric(create_test_tensor_file):
    """Test per-tensor symmetric quantization."""
    tensor_info = create_test_tensor_file
    config = {
        "Quantization Granularity": "Per-Tensor",
        "Quantization Type": "Symmetric",
        "Bit Width": "8-bit",
    }
    
    results = quantize_tensor(tensor_info, config)
    
    assert results is not None
    assert results["algorithm"] == "8-bit Per-Tensor Symmetric"
    assert results["quantized_dtype"] == "torch.qint8"
    assert "mse" in results
    assert "snr" in results

def test_per_tensor_asymmetric(create_test_tensor_file):
    """Test per-tensor asymmetric quantization."""
    tensor_info = create_test_tensor_file
    config = {
        "Quantization Granularity": "Per-Tensor",
        "Quantization Type": "Asymmetric",
        "Bit Width": "8-bit",
    }
    
    results = quantize_tensor(tensor_info, config)
    
    assert results is not None
    assert results["algorithm"] == "8-bit Per-Tensor Asymmetric"
    assert results["quantized_dtype"] == "torch.quint8"
    assert "scale" in results
    assert "zero_point" in results

def test_per_channel_symmetric(create_test_tensor_file):
    """Test per-channel symmetric quantization."""
    tensor_info = create_test_tensor_file
    config = {
        "Quantization Granularity": "Per-Channel",
        "Quantization Type": "Symmetric",
        "Bit Width": "8-bit",
    }
    
    results = quantize_tensor(tensor_info, config)
    
    assert results is not None
    assert results["algorithm"] == "8-bit Per-Channel Symmetric"
    assert "min:" in results["scale"]
    assert "min:" in results["zero_point"]

def test_per_group_symmetric(create_test_tensor_file):
    """Test per-group symmetric quantization."""
    tensor_info = create_test_tensor_file
    tensor_info["name"] = "1d_tensor"
    config = {
        "Quantization Granularity": "Per-Group",
        "Quantization Type": "Symmetric",
        "Bit Width": "8-bit",
        "Group Size": 10,
    }
    
    results = quantize_tensor(tensor_info, config)
    
    assert results is not None
    assert results["algorithm"] == "8-bit Per-Group Symmetric"

def test_per_block_symmetric(create_test_tensor_file):
    """Test per-block symmetric quantization."""
    tensor_info = create_test_tensor_file
    tensor_info["name"] = "block_tensor"
    config = {
        "Quantization Granularity": "Per-Block",
        "Quantization Type": "Symmetric",
        "Bit Width": "8-bit",
        "Block Size": 10,
    }
    
    results = quantize_tensor(tensor_info, config)
    
    assert results is not None
    assert results["algorithm"] == "8-bit Per-Block Symmetric"

def test_unsupported_dtype(create_test_tensor_file):
    """Test that non-float tensors raise an error."""
    tensor_info = create_test_tensor_file
    tensor_info["name"] = "int_tensor"
    config = {
        "Quantization Granularity": "Per-Tensor",
        "Quantization Type": "Symmetric",
    }
    
    with pytest.raises(QuantizationError, match="only supported for float tensors"):
        quantize_tensor(tensor_info, config)

def test_unsupported_granularity(create_test_tensor_file):
    """Test that unsupported granularities raise an error."""
    tensor_info = create_test_tensor_file
    config = {
        "Quantization Granularity": "Per-Foo",
        "Quantization Type": "Symmetric",
    }
    
    with pytest.raises(QuantizationError, match="not yet implemented"):
        quantize_tensor(tensor_info, config)

def test_unsupported_per_channel_asymmetric(create_test_tensor_file):
    """Test that per-channel asymmetric raises an error."""
    tensor_info = create_test_tensor_file
    config = {
        "Quantization Granularity": "Per-Channel",
        "Quantization Type": "Asymmetric",
    }
    
    with pytest.raises(QuantizationError, match="not supported by PyTorch"):
        quantize_tensor(tensor_info, config)

def test_unsupported_per_group_asymmetric(create_test_tensor_file):
    """Test that per-group asymmetric raises an error."""
    tensor_info = create_test_tensor_file
    config = {
        "Quantization Granularity": "Per-Group",
        "Quantization Type": "Asymmetric",
        "Group Size": 10,
    }
    
    with pytest.raises(QuantizationError, match="not supported"):
        quantize_tensor(tensor_info, config)

def test_unsupported_per_block_asymmetric(create_test_tensor_file):
    """Test that per-block asymmetric raises an error."""
    tensor_info = create_test_tensor_file
    tensor_info["name"] = "block_tensor"
    config = {
        "Quantization Granularity": "Per-Block",
        "Quantization Type": "Asymmetric",
        "Block Size": 10,
    }
    
    with pytest.raises(QuantizationError, match="not supported"):
        quantize_tensor(tensor_info, config)

def test_zero_tensor_quantization(create_test_tensor_file):
    """Test that a tensor of all zeros raises a specific error."""
    tensor_info = create_test_tensor_file
    tensor_info["name"] = "zero_tensor"
    config = {
        "Quantization Granularity": "Per-Tensor",
        "Quantization Type": "Symmetric",
    }

    with pytest.raises(QuantizationError, match="Cannot quantize tensor with all zero values"):
        quantize_tensor(tensor_info, config)

def test_per_group_not_divisible(create_test_tensor_file):
    """Test error when tensor elements not divisible by group size."""
    tensor_info = create_test_tensor_file
    tensor_info["name"] = "1d_tensor" # 100 elements
    config = {
        "Quantization Granularity": "Per-Group",
        "Quantization Type": "Symmetric",
        "Group Size": 9,
    }
    with pytest.raises(QuantizationError, match="must be divisible by Group Size"):
        quantize_tensor(tensor_info, config)

def test_per_block_not_divisible(create_test_tensor_file):
    """Test error when tensor dimensions not divisible by block size."""
    tensor_info = create_test_tensor_file
    tensor_info["name"] = "block_tensor" # 20x20
    config = {
        "Quantization Granularity": "Per-Block",
        "Quantization Type": "Symmetric",
        "Block Size": 9,
    }
    with pytest.raises(QuantizationError, match="must be divisible by Block Size"):
        quantize_tensor(tensor_info, config)

def test_per_block_not_2d(create_test_tensor_file):
    """Test error when using per-block on a non-2D tensor."""
    tensor_info = create_test_tensor_file
    tensor_info["name"] = "1d_tensor"
    config = {
        "Quantization Granularity": "Per-Block",
        "Quantization Type": "Symmetric",
        "Block Size": 10,
    }
    with pytest.raises(QuantizationError, match="only supported for 2D tensors"):
        quantize_tensor(tensor_info, config)