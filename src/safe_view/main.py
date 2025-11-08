import argparse
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Header, Footer, DataTable, Static, Button,
    Input, Label, SelectionList, TabbedContent,
    TabPane, Markdown, ProgressBar, RadioButton, RadioSet, OptionList
)
from textual_plotext import PlotextPlot
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
from textual.message import Message
import math

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
- **Sparsity**: {stats.get('sparsity', 0):.2f}%
"""
            if "preview" in stats:
                md_content += f"""\n### Data Preview\n\n```\n{stats['preview']}\n```\n"""

            if "quantiles" in stats and stats["quantiles"]:
                md_content += "\n### Quantile Distribution\n\n| Percentile | Value     |\n| :--- | :--- |\n"
                for p, v in stats["quantiles"]:
                    percentile_str = f"{p*100:.0f}%"
                    if p == 0.50:
                        percentile_str = f"**{p*100:.0f}% (Median)**"
                    md_content += f"| {percentile_str} | {v:<10.4f} |\n"
        else:
            # Show placeholder if statistics are not loaded yet
            md_content += f"""### Data Statistics
<Loading... Press Enter to load statistics>
"""

        self.update(md_content)


class TensorHistogramView(Static):
    """Displays a histogram of the tensor's values."""
    tensor_data = reactive(None)
    log_scale = reactive(True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_mount(self) -> None:
        self.query_one(ProgressBar).display = False

    def compose(self) -> ComposeResult:
        yield ProgressBar()
        yield PlotextPlot(id="plot")

    def show_progress(self):
        self.query_one(ProgressBar).display = True

    def hide_progress(self):
        self.query_one(ProgressBar).display = False

    def _render_plot(self, data: Dict) -> None:
        """Render the histogram plot based on the current data and scale."""
        plot_widget = self.query_one("#plot", PlotextPlot)
        plot = plot_widget.plt
        plot.clear_figure()

        if data and data.get("statistics") and "histogram" in data["statistics"]:
            hist_data = data["statistics"]["histogram"]
            values = hist_data["values"]
            bins = hist_data["bins"]
            x_labels = [f"{b:.2f}" for b in bins[:-1]]
            plot.plot_size(100, 20)

            if self.log_scale:
                epsilon = 0.1
                log_values = [
                    math.log(v) if v > 0 else epsilon for v in values]
                plot.bar(x_labels, log_values, orientation="v", width=0.1)
            else:
                plot.bar(x_labels, values, orientation="v", width=0.1)

            plot.title(f"Value Distribution for {data['name']}")
            plot.xlabel("Value Bins")
            plot.ylabel("Frequency")
        else:
            plot.title("Histogram")
            if data and data.get("name"):
                plot.bar(["Press 'x' to load"], [0])
            else:
                plot.bar(["No data"], [0])
            plot.xlabel("")
            plot.ylabel("")
        plot_widget.refresh()

    def watch_tensor_data(self, data: Dict) -> None:
        self.hide_progress()
        self._render_plot(data)

    def watch_log_scale(self, log_scale: bool) -> None:
        self._render_plot(self.tensor_data)


class QuantConfig:
    def __init__(self, name: str, description: str, value: Any = None,
                 options: List[str] = None, depends_on: List[str] = None, level: int = 0):
        self.name = name
        self.description = description
        self.value = value
        self.options = options or []
        self.depends_on = depends_on or []
        self.visible = True
        self.level = level


class QuantConfigScreen(Screen):

    BINDINGS = [
        ("j", "cursor_down", "Cursor Down"),
        ("k", "cursor_up", "Cursor Up"),
    ]

    CONFIG_OPTIONS = [
        QuantConfig("Bit Width", "选择量化位宽", "8-bit", ["8-bit", "4-bit"]),
        QuantConfig("Quantization Granularity", "选择量化粒度", "Per-Channel",
                    ["Per-Tensor", "Per-Channel", "Per-Group", "Per-Block"]),
        QuantConfig("Group Size", "分组量化大小", "128",
                    ["4", "8", "16", "32", "64", "128", "256"],
                    depends_on=["Per-Group", "Per-Block"], level=1),
        QuantConfig("Quantization Type", "量化类型", "Symmetric",
                    ["Symmetric", "Asymmetric"]),
        QuantConfig("Calibration Method", "校准方法", "Min-Max",
                    ["Min-Max", "KL Divergence"]),
    ]

    highlighted_index: int = 0

    def compose(self) -> ComposeResult:
        with Horizontal(classes="config-main"):
            with Vertical(classes="config-list-container"):
                yield Static("Quantization Configuration", classes="config-title")
                yield OptionList(id="config-list", classes="config-list")

            with Vertical(classes="detail-container"):
                yield Static("", id="option-title", classes="detail-title")
                yield Static("", id="option-description", classes="detail-desc")
                yield OptionList(id="value-options")
                with Horizontal(classes="button-group"):
                    yield Button("Select", id="select-btn", variant="primary")
                    yield Button("Back", id="back-btn", variant="default")

    def on_mount(self) -> None:
        self.update_visibility()
        self.query_one("#config-list", OptionList).highlighted = 0
        self.update_detail_view()
        self.query_one("#config-list").focus()

    @on(OptionList.OptionHighlighted, "#config-list")
    def on_config_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self.highlighted_index = event.option_index
        self.update_detail_view()

    @on(OptionList.OptionSelected, "#config-list")
    def on_config_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.query_one("#value-options").focus()

    @on(OptionList.OptionSelected, "#value-options")
    def on_value_options_option_selected(self, event: OptionList.OptionSelected) -> None:
        option = self.get_visible_options()[self.highlighted_index]
        option.value = option.options[event.option_index]
        self.update_visibility()
        self.query_one("#config-list").focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "select-btn":
            self.query_one("#value-options").focus()
        elif event.button.id == "back-btn":
            self.app.pop_screen()

    def get_visible_options(self) -> List[QuantConfig]:
        return [opt for opt in self.CONFIG_OPTIONS if opt.visible]

    def update_visibility(self) -> None:
        """根据依赖关系更新选项可见性"""
        granularity = next(
            opt for opt in self.CONFIG_OPTIONS if opt.name == "Quantization Granularity").value
        group_size_opt = next(
            opt for opt in self.CONFIG_OPTIONS if opt.name == "Group Size")

        group_size_opt.visible = granularity in group_size_opt.depends_on

        # 更新配置列表显示
        config_list = self.query_one("#config-list", OptionList)
        config_list.clear_options()
        
        visible_options = self.get_visible_options()
        for opt in visible_options:
            prefix = "  " * opt.level
            config_list.add_option(f"{prefix}{opt.name}")

        # 保持高亮位置
        if self.highlighted_index < len(visible_options):
            config_list.highlighted = self.highlighted_index
        else:
            config_list.highlighted = 0
        self.update_detail_view()

    def update_detail_view(self) -> None:
        """更新详情视图"""
        visible_options = self.get_visible_options()
        if self.highlighted_index >= len(visible_options):
            return

        option = visible_options[self.highlighted_index]

        self.query_one("#option-title", Static).update(option.name)
        self.query_one("#option-description",
                       Static).update(option.description)

        value_options = self.query_one("#value-options", OptionList)
        value_options.clear_options()
        if option.options:
            value_options.add_options(option.options)
            try:
                current_index = option.options.index(option.value)
                value_options.highlighted = current_index
            except ValueError:
                value_options.highlighted = 0
        
        # value_text = f"Current Value: {option.value}" if option.value else "Not Set"
        # self.query_one("#option-value", Static).update(value_text)

    def action_cursor_down(self) -> None:
        """Move cursor down in the focused option list."""
        config_list = self.query_one("#config-list", OptionList)
        value_options = self.query_one("#value-options", OptionList)

        if value_options.has_focus:
            if value_options.highlighted is not None and value_options.highlighted < len(value_options.options) - 1:
                value_options.highlighted += 1
        elif config_list.has_focus:
            if config_list.highlighted is not None and config_list.highlighted < len(config_list.options) - 1:
                config_list.highlighted += 1

    def action_cursor_up(self) -> None:
        """Move cursor up in the focused option list."""
        config_list = self.query_one("#config-list", OptionList)
        value_options = self.query_one("#value-options", OptionList)

        if value_options.has_focus:
            if value_options.highlighted is not None and value_options.highlighted > 0:
                value_options.highlighted -= 1
        elif config_list.has_focus:
            if config_list.highlighted is not None and config_list.highlighted > 0:
                config_list.highlighted -= 1


class SafeViewApp(App):
    class HistogramData(Message):
        """Message with histogram data."""

        def __init__(self, tensor_data: Dict):
            self.tensor_data = tensor_data
            super().__init__()

    """A terminal application to view safetensors files."""

    TITLE = "Safe View"
    CSS_PATH = "safe_view.css"
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("q", "quantize", "Quantize"),
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
        ("enter", "refresh_selected_tensor", "Refresh Selected Tensor"),
        ("ctrl+l", "toggle_log_scale", "Toggle Log Scale"),
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
                    with TabbedContent(initial="details-tab"):
                        with TabPane("Details", id="details-tab"):
                            yield TensorDetailView(id="tensor-detail")
                        with TabPane("Histogram", id="histogram-tab"):
                            yield TensorHistogramView(id="tensor-histogram")
                        with TabPane("Quantization", id="quantization-tab"):
                            yield Markdown("Select a quantization algorithm with 'q' to see results here.", id="quantization-detail")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.theme = "tokyo-night"
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
            histogram_view = self.query_one(TensorHistogramView)
            histogram_view.tensor_data = self.selected_tensor
            try:
                q_detail_view = self.query_one(
                    "#quantization-detail", Markdown)
                q_detail_view.update(
                    "Select a quantization algorithm with 'q' to see results here.")
            except:
                pass

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

    @on(Input.Submitted, "#search-input")
    def on_search_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search input submission."""
        if self.search_mode:
            self.filter_tensors(event.value)
        # Focus back on the table after search
        self.query_one(TensorInfoTable).focus()

    def _calculate_histogram_bins(self, tensor: torch.Tensor) -> int:
        """Calculate the optimal number of histogram bins using the Freedman-Diaconis rule."""
        n = tensor.numel()
        if n > 1:
            q1 = torch.quantile(tensor.float(), 0.25)
            q3 = torch.quantile(tensor.float(), 0.75)
            iqr = q3 - q1
            if iqr > 0:
                # Freedman-Diaconis rule
                # https://chat.deepseek.com/a/chat/s/41992b2b-3a17-4594-89d1-3f105fd69218
                bin_width = (2 * iqr) / (n ** (1/3))
                num_bins = int(
                    ((tensor.max() - tensor.min()) / bin_width).ceil().item())
            else:
                # Fallback for constant or quantized tensors (sqrt choice)
                num_bins = int(n ** 0.5)
        else:
            num_bins = 1
        # Cap the number of bins for display purposes
        num_bins = min(num_bins, 100)
        if num_bins == 0:
            num_bins = 1
        return num_bins

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
                "std": tensor.std().item(),
            }

            # Sparsity Calculation
            if tensor.numel() > 0:
                non_zero = torch.count_nonzero(tensor).item()
                stats["sparsity"] = (1 - (non_zero / tensor.numel())) * 100
            else:
                stats["sparsity"] = 0

            # Quantile Analysis
            if tensor.numel() > 0:
                quantile_points = torch.tensor(
                    [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
                quantile_values = torch.quantile(
                    tensor.float(), quantile_points)
                stats["quantiles"] = list(
                    zip(quantile_points.tolist(), quantile_values.tolist()))
            else:
                stats["quantiles"] = []

            # Data Preview Snippet
            # Use printoptions to safely truncate the tensor for preview
            torch.set_printoptions(
                precision=4, linewidth=80, sci_mode=False)
            if tensor.ndim == 0:
                preview_str = str(tensor.item())
            else:
                preview_str = str(tensor)
            stats["preview"] = preview_str
            torch.set_printoptions(profile="default")  # Reset to default

            # Calculate histogram
            num_bins = self._calculate_histogram_bins(tensor)
            values, bins = torch.histogram(tensor.float(), bins=num_bins)
            stats["histogram"] = {
                "values": values.tolist(),
                "bins": bins.tolist(),
            }

        # Create a new dictionary with the updated info
        new_tensor_info = tensor_info.copy()
        new_tensor_info["statistics"] = stats
        new_tensor_info["needs_loading"] = False

        return new_tensor_info

    @work(exclusive=True, group="stats", thread=True)
    def compute_statistics(self, tensor_info: Dict) -> None:
        """Worker to compute tensor statistics in the background."""
        try:
            updated_tensor = self.load_tensor_statistics(tensor_info)
            self.post_message(self.HistogramData(updated_tensor))
        except Exception as e:
            self.notify(
                f"Failed to load statistics: {str(e)}", severity="error")
            # Post back the original tensor_info on failure to ensure the UI updates and hides the progress bar
            self.post_message(self.HistogramData(tensor_info))

    def on_safe_view_app_histogram_data(self, message: HistogramData) -> None:
        """Called when histogram data is ready."""
        table = self.query_one(TensorInfoTable)
        if table.cursor_row is None:
            return

        updated_tensor = message.tensor_data
        self.filtered_tensors_data[table.cursor_row] = updated_tensor
        for i, tensor in enumerate(self.tensors_data):
            if tensor["name"] == updated_tensor["name"] and tensor["file_path"] == updated_tensor["file_path"]:
                self.tensors_data[i] = updated_tensor
                break

        detail_view = self.query_one(TensorDetailView)
        detail_view.tensor_data = updated_tensor
        histogram_view = self.query_one(TensorHistogramView)
        histogram_view.tensor_data = updated_tensor
        self.selected_tensor = updated_tensor

    def action_load_tensor_stats(self) -> None:
        """Load statistics for the currently selected tensor"""
        table = self.query_one(TensorInfoTable)
        if not self.filtered_tensors_data or table.cursor_row is None:
            return

        if 0 <= table.cursor_row < len(self.filtered_tensors_data):
            selected_tensor = self.filtered_tensors_data[table.cursor_row]

            if selected_tensor.get("needs_loading", False):
                histogram_view = self.query_one(TensorHistogramView)
                histogram_view._render_plot(None)
                histogram_view.show_progress()
                self.compute_statistics(selected_tensor)
            else:
                detail_view = self.query_one(TensorDetailView)
                detail_view.tensor_data = selected_tensor
                histogram_view = self.query_one(TensorHistogramView)
                histogram_view.tensor_data = selected_tensor
                self.selected_tensor = selected_tensor

    def action_refresh_selected_tensor(self) -> None:
        """Refresh the currently selected tensor (triggered by Enter key from anywhere)"""
        self.action_load_tensor_stats()

    @on(DataTable.RowSelected, "#tensor-table")
    def on_tensor_selected(self, event: DataTable.RowSelected) -> None:
        """Handles tensor selection events."""
        self.action_load_tensor_stats()

    def action_toggle_log_scale(self) -> None:
        """Toggle the log scale of the histogram."""
        self.query_one(TensorHistogramView).log_scale = not self.query_one(
            TensorHistogramView).log_scale

    def action_quantize(self) -> None:
        """Show quantization options."""
        if not self.selected_tensor:
            self.notify("No tensor selected.", severity="error")
            return
        self.load_tensor_statistics(self.selected_tensor)

        def quantization_callback(algorithm: str):
            if algorithm:
                self.quantize_tensor(algorithm)

        self.push_screen(QuantConfigScreen(), quantization_callback)

    def quantize_tensor(self, algorithm: str) -> None:
        """Quantize the selected tensor using the given algorithm."""
        tensor_info = self.selected_tensor

        with safe_open(tensor_info["file_path"], framework="pt", device="cpu") as f:
            tensor = f.get_tensor(tensor_info["name"])

        dtype = tensor.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            self.notify(
                f"Quantization only supported for float tensors. Selected tensor is {dtype}.", severity="error")
            return

        tensor = tensor.float()

        quantized_tensor = None
        if "per_tensor" in algorithm:
            if "symmetric" in algorithm:
                q_min, q_max = -128, 127
                scale = torch.max(torch.abs(tensor.min()),
                                  torch.abs(tensor.max())) / q_max
                zero_point = 0
                qdtype = torch.qint8
            else:
                q_min, q_max = 0, 255
                scale = (tensor.max() - tensor.min()) / (q_max - q_min)
                zero_point = q_min - torch.round(tensor.min() / scale)
                zero_point = int(zero_point.clamp(q_min, q_max).item())
                qdtype = torch.quint8

            if scale == 0:
                self.notify(
                    "Cannot quantize tensor with all zero values.", severity="warning")
                return

            quantized_tensor = torch.quantize_per_tensor(
                tensor, scale, zero_point, qdtype)

        elif "per_channel" in algorithm:
            if "asymmetric" in algorithm:
                self.notify(
                    "Per-channel asymmetric quantization is not supported by PyTorch.", severity="error")
                return

            if tensor.ndim <= 1:
                self.notify(
                    "Per-channel quantization requires at least 2 dimensions.", severity="error")
                return

            ch_axis = 0
            scales = torch.max(torch.abs(tensor.amin(dim=ch_axis, keepdim=True)), torch.abs(
                tensor.amax(dim=ch_axis, keepdim=True))) / 127
            scales = scales.flatten()
            scales[scales == 0] = 1.0
            zero_points = torch.zeros(scales.shape[0], dtype=torch.long)

            quantized_tensor = torch.quantize_per_channel(
                tensor, scales, zero_points, ch_axis, torch.qint8)

        if quantized_tensor is not None:
            dequantized_tensor = quantized_tensor.dequantize()
            mse = torch.mean((tensor - dequantized_tensor)**2)

            if mse > 0:
                snr = 10 * torch.log10(torch.mean(tensor**2) / mse)
                snr = snr.item()
            else:
                snr = float('inf')

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

            md_content = f"""## Quantization Results ({algorithm})

### Original Tensor
- **Data Type**: {str(dtype)}
- **Shape**: {' × '.join(map(str, tensor.shape))}
- **Min**: {tensor.min().item():.6f}
- **Max**: {tensor.max().item():.6f}

### Quantized Tensor
- **Data Type**: {str(quantized_tensor.dtype)}
- **Shape**: {' × '.join(map(str, quantized_tensor.shape))}
- **Scale**: {scale_str}
- **Zero Point**: {zp_str}

### Quality Metrics
- **Mean Squared Error (MSE)**: {mse.item():.6f}
- **Signal-to-Noise Ratio (SNR)**: {snr:.2f} dB
"""
            try:
                q_detail_view = self.query_one(
                    "#quantization-detail", Markdown)
                q_detail_view.update(md_content)
                self.query_one(TabbedContent).active = "quantization-tab"
            except Exception as e:
                self.notify(
                    f"Failed to update quantization tab: {e}", severity="error")


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