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
import torch
from typing import Dict, Any, List
import humanize
from huggingface_hub import snapshot_download

class SafetensorsHeader(Static):
    """显示safetensors文件头部信息"""

    def __init__(self):
        super().__init__()
        self.file_info = reactive({
            "filename": "",
            "file_size": 0,
            "tensor_count": 0,
            "total_parameters": 0
        })

    def render(self) -> str:
        if not self.file_info["filename"]:
            return "未选择文件"

        info = self.file_info
        return (
            f"文件: {info['filename']} | "
            f"大小: {info['file_size']/1024/1024:.2f} MB | "
            f"Tensor数量: {info['tensor_count']} | "
            f"总参数量: {info['total_parameters']:,}"
        )

class TensorInfoTable(DataTable):
    """显示tensors信息的表格"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cursor_type = "row"
        self.show_header = True
        self.zebra_stripes = True

    def on_mount(self) -> None:
        self.add_columns(
            "Tensor名称",
            "数据类型",
            "形状",
            "参数量",
            "大小(MB)"
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
    """显示选中tensor的详细信息"""
    tensor_data = reactive({})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def watch_tensor_data(self, data: Dict) -> None:
        self.log(f"watch_tensor_data received: {data}")
        if not data:
            self.update("选择左侧表格中的tensor查看详细信息")
            return

        # 计算统计信息
        shape_str = " × ".join(map(str, data["shape"]))
        total_elements = data["parameters"]

        md_content = f"""## Tensor: {data['name']}

### 基本信息
- **数据类型**: {data['dtype']}
- **形状**: {shape_str}
- **总元素数**: {total_elements:,}
- **内存占用**: {data['size_mb']:.2f} MB

### 数据统计
"""

        # 如果有实际数据，显示统计信息
        if "statistics" in data:
            stats = data["statistics"]
            md_content += f"""
- **最小值**: {stats.get('min', 'N/A'):.6f}
- **最大值**: {stats.get('max', 'N/A'):.6f}
- **平均值**: {stats.get('mean', 'N/A'):.6f}
- **标准差**: {stats.get('std', 'N/A'):.6f}
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
    ]

    def __init__(self, path: Path, title: str):
        super().__init__()
        self.path = path
        self.sub_title = title
        self.tensors_data = reactive([])
        self.selected_tensor = reactive({})
        self.filtered_tensors_data = reactive([])
        self.search_mode = False

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield SafetensorsHeader()
        with Container(id="app-body"):
            with Horizontal():
                with VerticalScroll(id="left-panel"):
                    yield Input(placeholder="输入tensor名称进行搜索... (按Escape退出)", id="search-input", classes="invisible")
                    yield TensorInfoTable(id="tensor-table")
                with VerticalScroll(id="right-panel"):
                    yield TensorDetailView(id="tensor-detail")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.title = "Safetensors文件查看器"
        self.sub_title = "可视化深度学习模型权重"
        self.process_safetensors_file()

    def on_ready(self) -> None:
        self.query_one(TensorInfoTable).focus()
        self.update_detail_view()

    def process_safetensors_file(self) -> None:
        """处理safetensors文件"""
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
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        shape = list(tensor.shape)
                        parameters = tensor.numel()
                        size_in_bytes = tensor.nelement() * tensor.element_size()
                        size_mb = size_in_bytes / 1024 / 1024
                        total_size += size_in_bytes

                        tensor_info = {
                            "name": key,
                            "dtype": str(tensor.dtype),
                            "shape": shape,
                            "parameters": parameters,
                            "size_mb": size_mb,
                            "tensor_object": tensor
                        }

                        if parameters <= 1_000_000_000:  # 限制计算数量
                            try:
                                tensor_info["statistics"] = {
                                    "min": tensor.min().item(),
                                    "max": tensor.max().item(),
                                    "mean": tensor.mean().item(),
                                    "std": tensor.std().item()
                                }
                            except:
                                pass

                        tensors_data.append(tensor_info)
                        total_parameters += parameters
            except Exception as e:
                self.notify(f"解析文件失败: {str(e)}", severity="error")
                return

        self.tensors_data = tensors_data
        self.filtered_tensors_data = tensors_data  # Initialize with all tensors

        header = self.query_one(SafetensorsHeader)
        header.file_info = {
            "filename": self.path.name,
            "file_size": total_size,
            "tensor_count": len(tensors_data),
            "total_parameters": total_parameters
        }

        table = self.query_one("#tensor-table", TensorInfoTable)
        table.update_table(self.filtered_tensors_data)

        self.notify(f"成功加载文件，包含 {len(tensors_data)} 个tensors")

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

    @on(DataTable.RowSelected, "#tensor-table")
    def on_tensor_selected(self, event: DataTable.RowSelected) -> None:
        """处理tensor选择事件"""
        if not self.filtered_tensors_data:
            return

        selected_row = event.cursor_row
        if 0 <= selected_row < len(self.filtered_tensors_data):
            self.selected_tensor = self.filtered_tensors_data[selected_row]
            self.log(f"on_tensor_selected: {self.selected_tensor['name']}")
            detail_view = self.query_one(TensorDetailView)
            detail_view.tensor_data = self.selected_tensor

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