import argparse
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    Header, Footer, DataTable, Static, Button,
    Input, Label, SelectionList, TabbedContent,
    TabPane, Markdown
)
from textual.reactive import reactive
import json
import struct
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import _getdtype
import torch
from typing import Dict, Any, List
import humanize
from huggingface_hub import snapshot_download

class SafetensorsHeader(Static):
    """Displays header information for the safetensors file."""

    file_info = reactive({
        "filename": "",
        "file_size": 0,
        "tensor_count": 0,
        "total_parameters": 0
    })

    def __init__(self):
        super().__init__()

    def render(self) -> str:
        if not self.file_info["filename"]:
            return "No file selected"

        info = self.file_info
        return (
            f"File: {info['filename']} | "
            f"Size: {info['file_size']/1024/1024:.2f} MB | "
            f"Tensors: {info['tensor_count']} | "
            f"Total Parameters: {info['total_parameters']:,}"
        )

class TensorInfoTable(DataTable):
    """A table to display tensor information."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cursor_type = "row"
        self.show_header = True
        self.zebra_stripes = True

    def on_mount(self) -> None:
        self.add_columns(
            "Tensor Name",
            "Data Type",
            "Shape",
            "Parameters",
            "Size (MB)"
        )

    def update_table(self, tensors_data: List[Dict]) -> None:
        self.clear()
        for tensor in tensors_data:
            shape_str = "×".join(map(str, tensor["shape"]))
            params = tensor["parameters"]
            size_mb = tensor["size_mb"]
            self.add_row(
                tensor["name"],
                tensor["dtype"],
                shape_str,
                f"{params:,}",
                f"{size_mb:.2f}MB"
            )

class TensorDetailView(Markdown):
    """Displays detailed information about the selected tensor."""
    tensor_data = reactive(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def watch_tensor_data(self, data: Dict) -> None:
        self.log(f"watch_tensor_data received: {data}")
        if not data:
            self.update(
                "Select a tensor from the table on the left to see details")
            return

        # Calculate statistics
        shape_str = " × ".join(map(str, data["shape"]))
        total_elements = data["parameters"]

        md_content = f"""## Tensor: {data['name']}

### Basic Information
- **Data Type**: {data['dtype']}
- **Shape**: {shape_str}
- **Total Elements**: {total_elements:,}
- **Memory Usage**: {data['size_mb']:.2f} MB

"""

        # Check if statistics are already loaded
        if data.get("statistics") is not None:
            stats = data["statistics"]
            md_content += f"""### Data Statistics
- **Min**: {stats.get('min', 'N/A'):.6f}
- **Max**: {stats.get('max', 'N/A'):.6f}
- **Mean**: {stats.get('mean', 'N/A'):.6f}
- **Std Dev**: {stats.get('std', 'N/A'):.6f}
"""
        else:
            # Show placeholder if statistics are not loaded yet
            md_content += f"""### Data Statistics
- **Min**: <Loading... Press Enter to load statistics>
- **Max**: <Loading... Press Enter to load statistics>
- **Mean**: <Loading... Press Enter to load statistics>
- **Std Dev**: <Loading... Press Enter to load statistics>
"""

        self.update(md_content)

class SafeViewApp(App):
    """A terminal application to view safetensors files."""

    TITLE = "Safe View"
    CSS_PATH = "safe_view.css"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("h", "scroll_left", "Scroll Left"),
        ("j", "cursor_down", "Cursor Down"),
        ("k", "cursor_up", "Cursor Up"),
        ("l", "scroll_right", "Scroll Right"),
        ("down", "cursor_down", "Cursor Down"),
        ("up", "cursor_up", "Cursor Up"),
        ("left", "scroll_left", "Scroll Left"),
        ("right", "scroll_right", "Scroll Right"),
        ("g", "go_to_top", "Go to Top"),
        ("G", "go_to_bottom", "Go to Bottom"),
        ("ctrl+f", "page_down", "Page Down"),
        ("ctrl+b", "page_up", "Page Up"),
        # ("ctrl+d", "half_page_down", "Half Page Down"),
        # ("ctrl+u", "half_page_up", "Half Page Up"),
        ("/", "search_tensor", "Search Tensor"),
        ("escape", "exit_search", "Exit Search"),
        ("x", "load_tensor_stats", "Load Tensor Statistics"),
    ]

    def __init__(self, path: Path, title: str):
        super().__init__()
        self.path = path
        self.sub_title = title
        self.tensors_data = []
        self.selected_tensor = {}
        self.filtered_tensors_data = []
        self.search_mode = False

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield SafetensorsHeader()
        with Container(id="app-body"):
            with Horizontal():
                with VerticalScroll(id="left-panel"):
                    yield Input(placeholder="Search for tensor name... (Press Escape to exit)", id="search-input", classes="invisible")
                    yield TensorInfoTable(id="tensor-table")
                with VerticalScroll(id="right-panel"):
                    yield TensorDetailView(id="tensor-detail")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.title = "Safetensors File Viewer"
        self.sub_title = "Visualize deep learning model weights"
        self.process_safetensors_file()

    def on_ready(self) -> None:
        self.query_one(TensorInfoTable).focus()
        self.update_detail_view()

    def process_safetensors_file(self) -> None:
        """Processes the safetensors file."""
        tensors_data = []
        total_parameters = 0
        total_size = 0

        files_to_process = []
        if self.path.is_file():
            files_to_process.append(self.path)
        else:
            files_to_process.extend(sorted(self.path.glob("**/*.safetensors")))

        for file_path in files_to_process:
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    # Get all tensor info from header without loading tensors
                    for key in f.keys():
                        # Get shape and dtype from the header info
                        shape = f.get_slice(key).get_shape()
                        dtype = f.get_slice(key).get_dtype()
                        dtype = _getdtype(dtype)

                        # Calculate parameters and size without loading tensor
                        parameters = 1
                        for dim in shape:
                            parameters *= dim
                        element_size = torch.tensor(
                            [], dtype=dtype).element_size()
                        size_in_bytes = parameters * element_size
                        size_mb = size_in_bytes / 1024 / 1024

                        total_size += size_in_bytes

                        tensor_info = {
                            "name": key,
                            "dtype": str(dtype),
                            "shape": list(shape),
                            "parameters": parameters,
                            "size_mb": size_mb,
                            "needs_loading": True,  # Flag to indicate tensor needs to be loaded for stats
                            "statistics": None,  # Will load on demand
                            # Store file path for later loading
                            "file_path": str(file_path)
                        }

                        tensors_data.append(tensor_info)
                        total_parameters += parameters
            except Exception as e:
                self.notify(
                    f"Failed to parse file: {str(e)}", severity="error")
                return

        self.tensors_data = tensors_data
        # Initialize with all tensors
        self.filtered_tensors_data = tensors_data[:]

        header = self.query_one(SafetensorsHeader)
        header.file_info = {
            "filename": self.path.name,
            "file_size": total_size,
            "tensor_count": len(tensors_data),
            "total_parameters": total_parameters
        }

        table = self.query_one("#tensor-table", TensorInfoTable)
        table.update_table(self.filtered_tensors_data)

        self.notify(
            f"Successfully loaded file with {len(tensors_data)} tensors")

    def update_detail_view(self) -> None:
        """Update the detail view with the selected tensor."""
        table = self.query_one(TensorInfoTable)
        self.log(f"update_detail_view called, cursor_row: {table.cursor_row}")
        if table.cursor_row is None:
            return
        if 0 <= table.cursor_row < len(self.filtered_tensors_data):
            self.selected_tensor = self.filtered_tensors_data[table.cursor_row]
            self.log(f"selected_tensor: {self.selected_tensor['name']}")
            detail_view = self.query_one(TensorDetailView)
            detail_view.tensor_data = self.selected_tensor

    def action_cursor_down(self) -> None:
        """Move cursor down in the table."""
        table = self.query_one(TensorInfoTable)
        if len(self.filtered_tensors_data) > 0:
            table.action_cursor_down()
            self.update_detail_view()

    def action_cursor_up(self) -> None:
        """Move cursor up in the table."""
        table = self.query_one(TensorInfoTable)
        if len(self.filtered_tensors_data) > 0:
            table.action_cursor_up()
            self.update_detail_view()

    def action_go_to_top(self) -> None:
        """Go to the top of the table."""
        table = self.query_one(TensorInfoTable)
        table.action_scroll_top()
        self.update_detail_view()

    def action_go_to_bottom(self) -> None:
        """Go to the bottom of the table."""
        table = self.query_one(TensorInfoTable)
        table.action_scroll_bottom()
        self.update_detail_view()

    def action_page_down(self) -> None:
        """Scroll down by a page."""
        table = self.query_one(TensorInfoTable)
        table.action_page_down()
        self.update_detail_view()

    def action_page_up(self) -> None:
        """Scroll up by a page."""
        table = self.query_one(TensorInfoTable)
        table.action_page_up()
        self.update_detail_view()

    def action_half_page_down(self) -> None:
        """Scroll down by half a page."""
        table = self.query_one(TensorInfoTable)
        scroll = self.query_one("#left-panel", VerticalScroll)
        table.action_cursor_down(scroll.window.height // 2)
        self.update_detail_view()

    def action_half_page_up(self) -> None:
        """Scroll up by half a page."""
        table = self.query_one(TensorInfoTable)
        scroll = self.query_one("#left-panel", VerticalScroll)
        table.action_cursor_up(scroll.window.height // 2)
        self.update_detail_view()


    def action_scroll_left(self) -> None:
        """Scroll left."""
        table = self.query_one(TensorInfoTable)
        table.action_cursor_left()

    def action_scroll_right(self) -> None:
        """Scroll right."""
        table = self.query_one(TensorInfoTable)
        table.action_cursor_right()

    def action_search_tensor(self) -> None:
        """Enter search mode."""
        self.search_mode = True
        search_input = self.query_one("#search-input", Input)
        search_input.visible = True
        search_input.focus()
        search_input.value = ""
        # self.notify("搜索模式已启用，输入tensor名称进行过滤", timeout=1)

    def action_exit_search(self) -> None:
        """Exit search mode and reset to full list."""
        self.search_mode = False
        search_input = self.query_one("#search-input", Input)
        search_input.visible = False
        search_input.value = ""
        self.filtered_tensors_data = self.tensors_data  # Reset to full list
        table = self.query_one("#tensor-table", TensorInfoTable)
        table.update_table(self.filtered_tensors_data)
        # self.notify("已退出搜索模式", timeout=1)
        self.query_one(TensorInfoTable).focus()

    def filter_tensors(self, search_term: str) -> None:
        """Filter tensors based on search term."""
        if not search_term:
            self.filtered_tensors_data = self.tensors_data
        else:
            # Filter tensors that contain the search term (case insensitive)
            filtered = [
                tensor for tensor in self.tensors_data
                if search_term.lower() in tensor["name"].lower()
            ]
            self.filtered_tensors_data = filtered

        # Update the table with filtered results
        table = self.query_one("#tensor-table", TensorInfoTable)
        table.update_table(self.filtered_tensors_data)

        # If there are results, focus on the first one
        if len(self.filtered_tensors_data) > 0:
            # table.cursor_row = 0
            self.update_detail_view()
        else:
            # Clear the detail view if no results
            detail_view = self.query_one(TensorDetailView)
            detail_view.tensor_data = {}

    @on(Input.Changed, "#search-input")
    def on_search_input_changed(self, event: Input.Changed) -> None:
        """Handle real-time search input changes."""
        if self.search_mode:
            self.filter_tensors(event.value)

    @on(Input.Submitted, "#search-input")
    def on_search_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search input submission."""
        # Focus back on the table after search
        self.query_one(TensorInfoTable).focus()

    def load_tensor_statistics(self, tensor_info: Dict) -> Dict:
        """Load tensor statistics on demand"""
        # Load the full tensor from the file
        with safe_open(tensor_info["file_path"], framework="pt", device="cpu") as f:
            tensor = f.get_tensor(tensor_info["name"])

            # Calculate statistics
            stats = {
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "mean": tensor.mean().item(),
                "std": tensor.std().item()
            }

        # Create a new dictionary with the updated info
        new_tensor_info = tensor_info.copy()
        new_tensor_info["statistics"] = stats
        new_tensor_info["needs_loading"] = False

        return new_tensor_info

    def action_load_tensor_stats(self) -> None:
        """Load statistics for the currently selected tensor"""
        table = self.query_one(TensorInfoTable)
        if not self.filtered_tensors_data or table.cursor_row is None:
            return

        if 0 <= table.cursor_row < len(self.filtered_tensors_data):
            self.log(
                f'action_load_tensor_stats called, cursor_row: {table.cursor_row}')
            selected_tensor = self.filtered_tensors_data[table.cursor_row]

            # Only load if we haven't loaded the statistics yet
            if selected_tensor.get("needs_loading", False):
                # self.notify(f"Loading statistics for {selected_tensor['name']}...")
                try:
                    updated_tensor = self.load_tensor_statistics(
                        selected_tensor)

                    # Update both the filtered data and the main data to keep them in sync
                    self.filtered_tensors_data[table.cursor_row] = updated_tensor
                    # Also update in the main tensors_data list
                    for i, tensor in enumerate(self.tensors_data):
                        if tensor["name"] == selected_tensor["name"] and tensor["file_path"] == selected_tensor["file_path"]:
                            self.tensors_data[i] = updated_tensor
                            break

                    # Update the detail view to show the new statistics
                    detail_view = self.query_one(TensorDetailView)
                    detail_view.tensor_data = updated_tensor
                    self.selected_tensor = updated_tensor
                    # self.notify(
                    #     f"Loaded statistics for {selected_tensor['name']}")
                except Exception as e:
                    self.notify(
                        f"Failed to load statistics: {str(e)}", severity="error")
            else:
                # Statistics already loaded, just update the view
                detail_view = self.query_one(TensorDetailView)
                detail_view.tensor_data = selected_tensor
                self.selected_tensor = selected_tensor

    @on(DataTable.RowSelected, "#tensor-table")
    def on_tensor_selected(self, event: DataTable.RowSelected) -> None:
        """Handles tensor selection events."""
        self.action_load_tensor_stats()

def main():
    """
    Display safetensors file information in a clean way.
    """
    parser = argparse.ArgumentParser(description="Safe View - A terminal application to view safetensors files")
    parser.add_argument("path", help="Path to a .safetensors file or a Hugging Face model ID")
    args = parser.parse_args()

    local_path = Path(args.path)
    title = args.path
    if not local_path.exists():
        try:
            local_path = Path(snapshot_download(repo_id=args.path, allow_patterns=["*.safetensors", "model.index.json"]))
        except Exception as e:
            print(f"Error downloading model from Hugging Face Hub: {e}")
            exit(1)

    app = SafeViewApp(local_path, title)
    app.run()

if __name__ == "__main__":
    main()