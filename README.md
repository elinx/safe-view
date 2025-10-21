# SafeView

A terminal application to view safetensors files. SafeView provides a clean, interactive terminal interface for exploring safetensors files and Hugging Face models.

## Features

- Interactive terminal UI for browsing tensors
- Detailed tensor information including shape, data type, and size
- Statistical information about tensor values (min, max, mean, std)
- Support for local safetensors files and Hugging Face model repositories
- Real-time search and filtering by tensor name
- Clean and intuitive Textual-based interface

## Installation

### Using pip:
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
- Click on a tensor in the left panel to view its details on the right

## Requirements

- Python 3.9+
- Dependencies listed in pyproject.toml