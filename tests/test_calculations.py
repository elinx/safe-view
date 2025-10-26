import pytest
import torch
from pathlib import Path
from safe_view.main import SafeViewApp

# Dummy path and title for app instantiation
dummy_path = Path("dummy.safetensors")
dummy_title = "dummy"

@pytest.fixture
def app_instance():
    """Provides an instance of SafeViewApp for testing."""
    # The app initializes some UI components that might require a running event loop.
    # We are only testing a calculation method, so we can patch the __init__ to avoid UI setup.
    app = SafeViewApp(path=dummy_path, title=dummy_title)
    return app

def test_freedman_diaconis_basic(app_instance):
    """Test with a standard normal distribution."""
    tensor = torch.randn(1000)
    num_bins = app_instance._calculate_histogram_bins(tensor)
    # For a standard normal tensor of size 1000, the FD rule gives ~23 bins.
    # We test for a reasonable range.
    assert isinstance(num_bins, int)
    assert 15 < num_bins < 35

def test_freedman_diaconis_constant_tensor(app_instance):
    """Test with a constant tensor where IQR is 0, should use the sqrt fallback."""
    tensor = torch.ones(100)
    num_bins = app_instance._calculate_histogram_bins(tensor)
    # sqrt(100) = 10
    assert num_bins == 10

def test_freedman_diaconis_single_element(app_instance):
    """Test with a single element tensor."""
    tensor = torch.tensor([5.0])
    num_bins = app_instance._calculate_histogram_bins(tensor)
    assert num_bins == 1

def test_freedman_diaconis_bin_capping(app_instance):
    """Test that the number of bins is capped at 100."""
    # A tensor with a very dense cluster and a few outliers will have a
    # small IQR but a large range, producing a huge number of bins.
    n = 100_000
    dense_cluster = torch.randn(n - 2) * 0.01  # Tiny IQR
    outliers = torch.tensor([-50.0, 50.0])      # Large range
    tensor = torch.cat([dense_cluster, outliers])
    num_bins = app_instance._calculate_histogram_bins(tensor)
    assert num_bins == 100

def test_freedman_diaconis_empty_tensor(app_instance):
    """Test with an empty tensor."""
    tensor = torch.tensor([])
    num_bins = app_instance._calculate_histogram_bins(tensor)
    assert num_bins == 1
