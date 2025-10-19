# SafeView

A terminal application to view safetensors files.

## Installation

1.  Install `uv`:

    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  Create a virtual environment:

    ```shell
    uv venv
    ```

3.  Activate the virtual environment:

    ```shell
    source .venv/bin/activate
    ```

4.  Install the dependencies:

    ```shell
    uv pip install .
    ```

## Usage

```shell
python -m safe_view.main /path/to/your/file.safetensors
```

Or for a Hugging Face model directory:

```shell
export TEXTUAL_LOG=./sv.log
export PYTHONPATH=./src:$PYTHONPATH
python -m safe_view.main Qwen/Qwen3-0.6B
```

