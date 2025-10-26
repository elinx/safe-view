# SafeView

A terminal application to view safetensors files. SafeView provides a clean, interactive terminal interface for exploring safetensors files and Hugging Face models.
![safe-view](./assets/screenshot_1.svg)
![safe-view search](./assets/screenshot_2.svg)

## Features

- Interactive terminal UI for browsing tensors
- Detailed tensor information including shape, data type, and size
- Statistical information about tensor values including min, max, mean, standard deviation, quantile analysis (1st, 10th, 25th, 50th/median, 75th, 90th, 99th percentiles), and sparsity percentage - loaded on demand when a tensor is selected
- Value distribution histogram visualization with toggle between linear and logarithmic scales
- Support for local safetensors files and Hugging Face model repositories with automatic downloading
- Real-time search and filtering by tensor name
- Clean and intuitive Textual-based interface with tabbed view (Details and Histogram)
- Optimized loading - only metadata is loaded initially, tensor statistics shown when a tensor is selected
- Progress indicators when loading tensor statistics
- Data preview snippets for quick inspection
- File-level information including total size, tensor count, and total parameter count

## Installation

### From PyPI:
```shell
pip install safe-view
```

### Using pip (from local directory):
```shell
pip install .
```

### Using uv:
```shell
uv pip install .
```

### Development mode:
If you want to run in development mode, you can install in editable mode:

```shell
pip install -e .
```

or with uv:

```shell
uv pip install -e .
```

## Usage

After installation, you can run the application directly from the command line:

```shell
safe-view /path/to/your/file.safetensors
```

Or for a Hugging Face model:

```shell
safe-view Qwen/Qwen3-0.6B
```

For help:

```shell
safe-view --help
```

## Controls

- `q`: Quit the application
- `h`, `j`, `k`, `l` or arrow keys: Navigate between tensors
- `g`: Go to top of the tensor list
- `G`: Go to bottom of the tensor list
- `Ctrl+f` / `Ctrl+b`: Page up/down
- `/`: Enter search mode to filter tensors by name
- `Escape`: Exit search mode
- `x` or `Enter`: Load and display detailed statistics for the selected tensor
- `Ctrl+l`: Toggle histogram between linear and logarithmic scale
- Click on a tensor in the left panel to view its details and statistics on the right

## Requirements

- Python 3.9+
- Dependencies listed in pyproject.toml