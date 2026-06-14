from __future__ import annotations

import json
from pathlib import Path

import wx
import wx.html2

from ..ai_cleanup import (
    LlamaCppBackend,
    OpenAICompatBackend,
    find_ai_cleanup_blocks,
    is_llama_cpp_available,
    run_ai_cleanup,
)
from ..db import CycleError, NodeNotFoundError, Repository, RepositoryError
from ..exporter import export_tree
from ..importer import import_folder
from ..markdown_render import render_markdown

SELECTED_NODE_META_KEY = "ui.selected_node_id"
EXPANDED_NODES_META_KEY = "ui.expanded_node_ids"
TREE_TOP_NODE_META_KEY = "ui.tree_top_node_id"
EDITOR_POSITIONS_META_KEY = "ui.editor_positions"
VIEW_POSITIONS_META_KEY = "ui.view_positions"
THEME_META_KEY = "ui.theme"
AI_MODEL_PATH_META_KEY = "ai.model_path"
AI_BACKEND_META_KEY = "ai.backend"
AI_OPENAI_BASE_URL_META_KEY = "ai.openai.base_url"
AI_OPENAI_API_KEY_META_KEY = "ai.openai.api_key"
AI_OPENAI_MODEL_META_KEY = "ai.openai.model"
AI_OPENAI_MAX_TOKENS_META_KEY = "ai.openai.max_tokens"
AI_OPENAI_TEMPERATURE_META_KEY = "ai.openai.temperature"
AI_OPENAI_TOP_P_META_KEY = "ai.openai.top_p"
AI_OPENAI_FREQUENCY_PENALTY_META_KEY = "ai.openai.frequency_penalty"
AI_OPENAI_PRESENCE_PENALTY_META_KEY = "ai.openai.presence_penalty"
AI_OPENAI_TIMEOUT_META_KEY = "ai.openai.timeout_seconds"
AI_LOCAL_MAX_TOKENS_META_KEY = "ai.local.max_tokens"
AI_LOCAL_TEMPERATURE_META_KEY = "ai.local.temperature"
AI_LOCAL_REPEAT_PENALTY_META_KEY = "ai.local.repeat_penalty"
AI_LOCAL_TOP_K_META_KEY = "ai.local.top_k"
AI_LOCAL_TOP_P_META_KEY = "ai.local.top_p"
AI_LOCAL_MIN_P_META_KEY = "ai.local.min_p"
AI_LOCAL_FREQUENCY_PENALTY_META_KEY = "ai.local.frequency_penalty"
AI_LOCAL_PRESENCE_PENALTY_META_KEY = "ai.local.presence_penalty"
AI_LOCAL_SEED_META_KEY = "ai.local.seed"
AI_LOCAL_CONTEXT_META_KEY = "ai.local.n_ctx"
AI_LOCAL_THREADS_META_KEY = "ai.local.n_threads"
AI_LOCAL_GPU_LAYERS_META_KEY = "ai.local.n_gpu_layers"
AI_LOCAL_THINKING_MODE_META_KEY = "ai.local.thinking_mode"
AI_CONTEXT_CHARS_META_KEY = "ai.context_chars"
AI_LOCAL_BATCH_META_KEY = "ai.local.n_batch"
AI_LOCAL_UBATCH_META_KEY = "ai.local.n_ubatch"
AI_LOCAL_OFFLOAD_KQV_META_KEY = "ai.local.offload_kqv"
AI_LOCAL_FLASH_ATTN_META_KEY = "ai.local.flash_attn"

AI_BACKEND_LOCAL = "local"
AI_BACKEND_OPENAI = "openai"
AI_THINKING_DISABLED = "disabled"
AI_THINKING_ALLOWED = "allowed"
SPLITTER_SASH_SIZE = 10

THEMES = {
    "light": {
        "background": "#f5f6f8",
        "panel": "#ffffff",
        "foreground": "#202124",
        "editor_background": "#ffffff",
        "editor_foreground": "#202124",
        "tree_background": "#ffffff",
        "tree_foreground": "#202124",
    },
    "dark": {
        "background": "#1f2328",
        "panel": "#252a31",
        "foreground": "#e6edf3",
        "editor_background": "#1f2328",
        "editor_foreground": "#e6edf3",
        "tree_background": "#252a31",
        "tree_foreground": "#e6edf3",
    },
}


class MainFrame(wx.Frame):
    def __init__(self, parent: wx.Window | None, repo: Repository, db_path: Path):
        super().__init__(parent, title="Portable Markdown Editor", size=(1100, 760))
        self.repo = repo
        self.db_path = db_path
        self.selected_id: int | None = self._load_persisted_selected_id()
        self.dirty = False
        self.dragged_id: int | None = None
        self.restoring_tree = False
        self.loading_node = False
        self.pending_view_scroll_y = 0
        self.theme = self._load_theme()
        self.ai_backend_type = self._load_ai_backend_type()
        self.ai_backend: object | None = None
        self.local_backend_signature: tuple[str, ...] | None = None
        self.llm_model_path: Path | None = self._load_ai_model_path()
        self.last_ai_undo_text: str | None = None
        self.editor_selection_click: tuple[wx.Point, int] | None = None
        self.autosave_timer = wx.Timer(self)

        self._build_menu()
        self._build_ui()
        self._bind_events()
        self.apply_theme()
        self.refresh_tree()
        self.load_selected_node()
        self.SetStatusText(str(db_path))

    def _build_menu(self) -> None:
        menu_bar = wx.MenuBar()
        file_menu = wx.Menu()
        self.new_library_item = file_menu.Append(wx.ID_NEW, "New Library...\tCtrl+N")
        self.open_library_item = file_menu.Append(wx.ID_OPEN, "Open Library...\tCtrl+O")
        file_menu.AppendSeparator()
        self.import_item = file_menu.Append(wx.ID_ANY, "Import Folder...")
        self.export_selected_item = file_menu.Append(wx.ID_ANY, "Export Selected...")
        self.export_all_item = file_menu.Append(wx.ID_ANY, "Export All...")
        file_menu.AppendSeparator()
        self.exit_item = file_menu.Append(wx.ID_EXIT, "Exit")
        menu_bar.Append(file_menu, "File")
        view_menu = wx.Menu()
        self.night_mode_item = view_menu.AppendCheckItem(wx.ID_ANY, "Night Mode")
        self.night_mode_item.Check(self.theme == "dark")
        menu_bar.Append(view_menu, "View")
        ai_menu = wx.Menu()
        self.ai_settings_item = ai_menu.Append(wx.ID_ANY, "Settings...")
        self.select_ai_model_item = ai_menu.Append(wx.ID_ANY, "Select Model...")
        self.select_ai_model_item.Enable(is_llama_cpp_available())
        self.ai_cleanup_menu_item = ai_menu.Append(wx.ID_ANY, "Cleanup Marked Blocks")
        menu_bar.Append(ai_menu, "AI")
        self.SetMenuBar(menu_bar)
        self.CreateStatusBar()

    def _build_ui(self) -> None:
        self.splitter = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        self._set_splitter_sash_size(self.splitter, SPLITTER_SASH_SIZE)

        self.tree_panel = wx.Panel(self.splitter)
        tree_sizer = wx.BoxSizer(wx.VERTICAL)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.add_root_btn = wx.Button(self.tree_panel, label="Add Root")
        self.add_child_btn = wx.Button(self.tree_panel, label="Add Child")
        self.rename_btn = wx.Button(self.tree_panel, label="Rename")
        self.delete_btn = wx.Button(self.tree_panel, label="Delete")
        for button in (self.add_root_btn, self.add_child_btn, self.rename_btn, self.delete_btn):
            button_sizer.Add(button, 0, wx.ALL, 4)
        self.tree = wx.TreeCtrl(self.tree_panel, style=wx.TR_HAS_BUTTONS | wx.TR_LINES_AT_ROOT | wx.TR_DEFAULT_STYLE)
        tree_sizer.Add(button_sizer, 0, wx.EXPAND)
        tree_sizer.Add(self.tree, 1, wx.EXPAND | wx.ALL, 4)
        self.tree_panel.SetSizer(tree_sizer)

        self.content_notebook = wx.Notebook(self.splitter)

        self.edit_panel = wx.Panel(self.content_notebook)
        edit_sizer = wx.BoxSizer(wx.VERTICAL)
        edit_toolbar = wx.BoxSizer(wx.HORIZONTAL)
        self.title_ctrl = wx.TextCtrl(self.edit_panel)
        self._disable_smart_text_substitutions(self.title_ctrl)
        self.save_btn = wx.Button(self.edit_panel, label="Save")
        self.ai_cleanup_btn = wx.Button(self.edit_panel, label="AI Cleanup")
        self.undo_ai_btn = wx.Button(self.edit_panel, label="Undo AI")
        self.undo_ai_btn.Enable(False)
        edit_toolbar.Add(wx.StaticText(self.edit_panel, label="Title"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        edit_toolbar.Add(self.title_ctrl, 1, wx.RIGHT, 6)
        edit_toolbar.Add(self.save_btn, 0)
        edit_toolbar.Add(self.ai_cleanup_btn, 0, wx.LEFT, 6)
        edit_toolbar.Add(self.undo_ai_btn, 0, wx.LEFT, 6)
        self.editor = wx.TextCtrl(self.edit_panel, style=wx.TE_MULTILINE | wx.TE_RICH2 | wx.TE_PROCESS_TAB)
        self._disable_smart_text_substitutions(self.editor)
        edit_sizer.Add(edit_toolbar, 0, wx.EXPAND | wx.ALL, 6)
        edit_sizer.Add(self.editor, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        self.edit_panel.SetSizer(edit_sizer)

        self.view_panel = wx.Panel(self.content_notebook)
        view_sizer = wx.BoxSizer(wx.VERTICAL)
        self.preview = wx.html2.WebView.New(self.view_panel)
        view_sizer.Add(self.preview, 1, wx.EXPAND)
        self.view_panel.SetSizer(view_sizer)

        self.content_notebook.AddPage(self.edit_panel, "Edit")
        self.content_notebook.AddPage(self.view_panel, "View")

        self.splitter.SplitVertically(self.tree_panel, self.content_notebook, 320)
        self.splitter.SetMinimumPaneSize(220)

    def _bind_events(self) -> None:
        self.Bind(wx.EVT_MENU, self.on_new_library, self.new_library_item)
        self.Bind(wx.EVT_MENU, self.on_open_library, self.open_library_item)
        self.Bind(wx.EVT_MENU, self.on_import_folder, self.import_item)
        self.Bind(wx.EVT_MENU, self.on_export_selected, self.export_selected_item)
        self.Bind(wx.EVT_MENU, self.on_export_all, self.export_all_item)
        self.Bind(wx.EVT_MENU, self.on_toggle_night_mode, self.night_mode_item)
        self.Bind(wx.EVT_MENU, self.on_ai_settings, self.ai_settings_item)
        self.Bind(wx.EVT_MENU, self.on_select_ai_model, self.select_ai_model_item)
        self.Bind(wx.EVT_MENU, self.on_ai_cleanup, self.ai_cleanup_menu_item)
        self.Bind(wx.EVT_MENU, lambda event: self.Close(), self.exit_item)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Bind(wx.EVT_TIMER, self.on_autosave_timer, self.autosave_timer)

        self.tree.Bind(wx.EVT_TREE_SEL_CHANGED, self.on_tree_selection)
        self.tree.Bind(wx.EVT_TREE_ITEM_EXPANDED, self.on_tree_expansion_changed)
        self.tree.Bind(wx.EVT_TREE_ITEM_COLLAPSED, self.on_tree_expansion_changed)
        self.tree.Bind(wx.EVT_TREE_BEGIN_DRAG, self.on_begin_drag)
        self.tree.Bind(wx.EVT_TREE_END_DRAG, self.on_end_drag)
        self.add_root_btn.Bind(wx.EVT_BUTTON, self.on_add_root)
        self.add_child_btn.Bind(wx.EVT_BUTTON, self.on_add_child)
        self.rename_btn.Bind(wx.EVT_BUTTON, self.on_rename)
        self.delete_btn.Bind(wx.EVT_BUTTON, self.on_delete)
        self.save_btn.Bind(wx.EVT_BUTTON, self.on_save)
        self.ai_cleanup_btn.Bind(wx.EVT_BUTTON, self.on_ai_cleanup)
        self.undo_ai_btn.Bind(wx.EVT_BUTTON, self.on_undo_ai)
        self.editor.Bind(wx.EVT_TEXT, self.on_text_changed)
        self.editor.Bind(wx.EVT_KEY_DOWN, self.on_editor_key_down)
        self.editor.Bind(wx.EVT_LEFT_DOWN, self.on_editor_left_down)
        self.editor.Bind(wx.EVT_LEFT_UP, self.on_editor_left_up)
        self.title_ctrl.Bind(wx.EVT_TEXT, self.on_text_changed)
        self.content_notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_page_changed)
        self.preview.Bind(wx.html2.EVT_WEBVIEW_LOADED, self.on_preview_loaded)

    def refresh_tree(self) -> None:
        expanded_ids = self._load_persisted_expanded_ids()
        self.restoring_tree = True
        self.tree.Freeze()
        try:
            self.tree.DeleteAllItems()
            root = self.tree.AddRoot("Library")
            self.tree.SetItemData(root, None)
            self._append_children(root, None)
            self.tree.Expand(root)
            self._restore_expanded_items(root, expanded_ids)
            if self.selected_id is not None and not self._select_node_item(root, self.selected_id):
                self.selected_id = None
            self._restore_tree_top_visible_id()
        finally:
            self.tree.Thaw()
            self.restoring_tree = False
        self._update_button_state()

    def _append_children(self, parent_item: wx.TreeItemId, parent_id: int | None) -> None:
        for node in self.repo.list_children(parent_id):
            item = self.tree.AppendItem(parent_item, node.title)
            self.tree.SetItemData(item, node.id)
            self._append_children(item, node.id)

    def _select_node_item(self, item: wx.TreeItemId, node_id: int) -> bool:
        if self.tree.GetItemData(item) == node_id:
            self.tree.SelectItem(item)
            self.tree.EnsureVisible(item)
            return True
        child, cookie = self.tree.GetFirstChild(item)
        while child.IsOk():
            if self._select_node_item(child, node_id):
                self.tree.Expand(item)
                return True
            child, cookie = self.tree.GetNextChild(item, cookie)
        return False

    def _restore_expanded_items(self, item: wx.TreeItemId, expanded_ids: set[int]) -> None:
        node_id = self.tree.GetItemData(item)
        if node_id in expanded_ids:
            self.tree.Expand(item)
        child, cookie = self.tree.GetFirstChild(item)
        while child.IsOk():
            self._restore_expanded_items(child, expanded_ids)
            child, cookie = self.tree.GetNextChild(item, cookie)

    def load_selected_node(self) -> None:
        self.loading_node = True
        self.autosave_timer.Stop()
        try:
            if self.selected_id is None:
                self.title_ctrl.ChangeValue("")
                self.editor.ChangeValue("")
                self.preview.SetPage(render_markdown("", self.theme), "")
                self.dirty = False
                return
            node = self.repo.get_node(self.selected_id)
            self.title_ctrl.ChangeValue(node.title)
            self.editor.ChangeValue(node.markdown_content)
            self._restore_editor_position(node.id)
            self.pending_view_scroll_y = self._load_view_scroll_y(node.id)
            self.preview.SetPage(render_markdown(node.markdown_content, self.theme), "")
            self.dirty = False
            self._update_button_state()
        finally:
            self.loading_node = False

    def save_current_node(self) -> None:
        if self.selected_id is None or not self.dirty:
            return
        self.autosave_timer.Stop()
        title = self.title_ctrl.GetValue().strip()
        if not title:
            self._show_error("Title cannot be blank.")
            return
        content = self.editor.GetValue()
        node = self.repo.update_node(self.selected_id, title=title, content=content)
        selected_item = self.tree.GetSelection()
        if selected_item.IsOk() and self.tree.GetItemData(selected_item) == node.id:
            self.tree.SetItemText(selected_item, node.title)
        if self._current_content_page_label() == "View":
            self.pending_view_scroll_y = self._load_view_scroll_y(node.id)
            self.preview.SetPage(render_markdown(node.markdown_content, self.theme), "")
        self.dirty = False

    def on_tree_selection(self, event: wx.TreeEvent) -> None:
        if self.restoring_tree:
            event.Skip()
            return
        self._persist_current_positions()
        self.save_current_node()
        item = event.GetItem()
        self.selected_id = self.tree.GetItemData(item)
        self.load_selected_node()
        if not self.restoring_tree:
            self._persist_selected_node()
            self._persist_tree_top_visible_id()

    def on_begin_drag(self, event: wx.TreeEvent) -> None:
        node_id = self.tree.GetItemData(event.GetItem())
        if node_id is not None:
            self.dragged_id = node_id
            event.Allow()

    def on_end_drag(self, event: wx.TreeEvent) -> None:
        if self.dragged_id is None:
            return
        target = event.GetItem()
        new_parent_id = self.tree.GetItemData(target) if target.IsOk() else None
        try:
            self.repo.move_node(self.dragged_id, new_parent_id)
            self.selected_id = self.dragged_id
            self.refresh_tree()
            self._persist_selected_node()
            self._persist_expanded_nodes()
            self._persist_tree_top_visible_id()
        except CycleError as exc:
            self._show_error(str(exc))
        finally:
            self.dragged_id = None

    def on_add_root(self, event: wx.CommandEvent) -> None:
        self._create_node(None)

    def on_add_child(self, event: wx.CommandEvent) -> None:
        self._create_node(self.selected_id)

    def _create_node(self, parent_id: int | None) -> None:
        with wx.TextEntryDialog(self, "Node title", "Create Node", "Untitled") as dialog:
            if dialog.ShowModal() != wx.ID_OK:
                return
            node = self.repo.create_node(parent_id, dialog.GetValue())
            self.selected_id = node.id
            self.refresh_tree()
            self.load_selected_node()
            self._persist_selected_node()
            self._persist_expanded_nodes()
            self._persist_tree_top_visible_id()

    def on_rename(self, event: wx.CommandEvent) -> None:
        if self.selected_id is None:
            return
        node = self.repo.get_node(self.selected_id)
        with wx.TextEntryDialog(self, "Node title", "Rename Node", node.title) as dialog:
            if dialog.ShowModal() != wx.ID_OK:
                return
            self.repo.update_node(node.id, title=dialog.GetValue())
            self.refresh_tree()
            self.load_selected_node()
            self._persist_selected_node()
            self._persist_expanded_nodes()
            self._persist_tree_top_visible_id()

    def on_delete(self, event: wx.CommandEvent) -> None:
        if self.selected_id is None:
            return
        node = self.repo.get_node(self.selected_id)
        message = f"Delete '{node.title}' and all child nodes?"
        if wx.MessageBox(message, "Delete Node", wx.YES_NO | wx.ICON_WARNING) != wx.YES:
            return
        self.repo.delete_node(node.id, cascade=True)
        self.selected_id = None
        self.refresh_tree()
        self.load_selected_node()
        self._persist_selected_node()
        self._persist_expanded_nodes()
        self._persist_tree_top_visible_id()

    def on_save(self, event: wx.CommandEvent) -> None:
        self.save_current_node()

    def on_text_changed(self, event: wx.CommandEvent) -> None:
        if self.selected_id is not None and not self.loading_node:
            self.dirty = True
            self.autosave_timer.StartOnce(1000)
        event.Skip()

    def on_autosave_timer(self, event: wx.TimerEvent) -> None:
        if self.selected_id is None or not self.dirty:
            return
        focused = wx.Window.FindFocus()
        if focused not in (self.editor, self.title_ctrl):
            return
        self.save_current_node()

    def on_tree_expansion_changed(self, event: wx.TreeEvent) -> None:
        if not self.restoring_tree:
            self._persist_expanded_nodes()
            self._persist_tree_top_visible_id()
        event.Skip()

    def on_editor_key_down(self, event: wx.KeyEvent) -> None:
        if event.GetKeyCode() != wx.WXK_TAB:
            event.Skip()
            return
        if event.ShiftDown():
            self._unindent_selection()
        else:
            self._indent_selection()
        self.dirty = self.selected_id is not None

    def on_editor_left_down(self, event: wx.MouseEvent) -> None:
        self.editor_selection_click = None
        start, end = self.editor.GetSelection()
        if start != end and not (event.ShiftDown() or event.ControlDown() or event.AltDown()):
            position = self._editor_position_from_point(event.GetPosition())
            if position is not None:
                self.editor_selection_click = (event.GetPosition(), position)
        event.Skip()

    def on_editor_left_up(self, event: wx.MouseEvent) -> None:
        pending_click = self.editor_selection_click
        self.editor_selection_click = None
        if pending_click is not None:
            down_point, position = pending_click
            up_point = event.GetPosition()
            if abs(up_point.x - down_point.x) <= 3 and abs(up_point.y - down_point.y) <= 3:
                wx.CallAfter(self._collapse_editor_selection, position)
        event.Skip()

    def _editor_position_from_point(self, point: wx.Point) -> int | None:
        result, column, row = self.editor.HitTest(point)
        if result == wx.TE_HT_UNKNOWN:
            return None
        try:
            position = self.editor.XYToPosition(column, row)
        except wx.wxAssertionError:
            return None
        return position if position >= 0 else None

    def _collapse_editor_selection(self, position: int) -> None:
        start, end = self.editor.GetSelection()
        if start != end:
            self.editor.SetInsertionPoint(position)

    def _indent_selection(self) -> None:
        start, end = self.editor.GetSelection()
        if start == end:
            self.editor.WriteText("\t")
            return
        text = self.editor.GetValue()
        line_start = text.rfind("\n", 0, start) + 1
        selected = text[line_start:end]
        replacement = "\t" + selected.replace("\n", "\n\t")
        self.editor.Replace(line_start, end, replacement)
        self.editor.SetSelection(line_start, line_start + len(replacement))

    def _unindent_selection(self) -> None:
        start, end = self.editor.GetSelection()
        text = self.editor.GetValue()
        line_start = text.rfind("\n", 0, start) + 1
        line_end = end if start != end else text.find("\n", start)
        if line_end == -1:
            line_end = len(text)
        selected = text[line_start:line_end]
        lines = selected.split("\n")
        changed = False
        for index, line in enumerate(lines):
            if line.startswith("\t"):
                lines[index] = line[1:]
                changed = True
            elif line.startswith("    "):
                lines[index] = line[4:]
                changed = True
        if not changed:
            return
        replacement = "\n".join(lines)
        self.editor.Replace(line_start, line_end, replacement)
        self.editor.SetSelection(line_start, line_start + len(replacement))

    def on_page_changed(self, event: wx.BookCtrlEvent) -> None:
        old_selection = event.GetOldSelection()
        if old_selection != wx.NOT_FOUND:
            old_label = self.content_notebook.GetPageText(old_selection)
            if old_label in {"Edit", "View"}:
                self._persist_current_positions(page_label=old_label)
        if self.content_notebook.GetPageText(event.GetSelection()) == "View":
            self.save_current_node()
            if self.selected_id is not None:
                self.pending_view_scroll_y = self._load_view_scroll_y(self.selected_id)
                self.preview.SetPage(render_markdown(self.repo.get_node(self.selected_id).markdown_content, self.theme), "")
        event.Skip()

    def on_preview_loaded(self, event: wx.html2.WebViewEvent) -> None:
        if self.pending_view_scroll_y > 0:
            self._set_view_scroll_y(self.pending_view_scroll_y)
        event.Skip()

    def on_toggle_night_mode(self, event: wx.CommandEvent) -> None:
        self.theme = "dark" if self.night_mode_item.IsChecked() else "light"
        self.repo.set_meta(THEME_META_KEY, self.theme)
        self.apply_theme()
        self._rerender_preview()

    def on_select_ai_model(self, event: wx.CommandEvent) -> None:
        if not is_llama_cpp_available():
            self._show_error("llama-cpp-python is not installed. Local llama.cpp models are unavailable.")
            return
        if self._prompt_ai_model_path():
            self.ai_backend_type = AI_BACKEND_LOCAL
            self.ai_backend = None
            self.repo.set_meta(AI_BACKEND_META_KEY, self.ai_backend_type)

    def on_ai_settings(self, event: wx.CommandEvent) -> None:
        old_backend_type = self.ai_backend_type
        old_signature = self._local_backend_signature(self._load_ai_settings())
        with AISettingsDialog(self, self._load_ai_settings()) as dialog:
            if dialog.ShowModal() != wx.ID_OK:
                return
            settings = dialog.get_settings()
        self._save_ai_settings(settings)
        self.ai_backend_type = settings["backend"]
        self.llm_model_path = Path(settings["local_model_path"]) if settings["local_model_path"] else None
        new_signature = self._local_backend_signature(settings)
        if isinstance(self.ai_backend, LlamaCppBackend):
            if self.ai_backend_type == AI_BACKEND_LOCAL and new_signature != old_signature:
                wx.MessageBox(
                    "Local llama.cpp settings were saved, but the already-loaded CUDA model will keep using "
                    "the previous settings until the app is restarted. This avoids a known CUDA reload crash.",
                    "AI Settings",
                    wx.OK | wx.ICON_INFORMATION,
                )
            elif old_backend_type != self.ai_backend_type:
                wx.MessageBox(
                    "AI backend setting was saved. Restart the app before loading another local llama.cpp model "
                    "in this session to avoid CUDA reload errors.",
                    "AI Settings",
                    wx.OK | wx.ICON_INFORMATION,
                )
        else:
            self.ai_backend = None
            self.local_backend_signature = None

    def on_ai_cleanup(self, event: wx.CommandEvent) -> None:
        if self.selected_id is None:
            self._show_error("Select a document before running AI cleanup.")
            return
        content = self.editor.GetValue()
        blocks = find_ai_cleanup_blocks(content)
        if not blocks:
            self._show_error("No AI cleanup blocks found. Use <!-- AI: instruction --> ... <!-- AI: end -->.")
            return
        backend = self._get_ai_backend()
        if backend is None:
            return
        try:
            settings = self._load_ai_settings()
            with wx.BusyInfo(f"Running AI cleanup for {len(blocks)} block(s)..."):
                result = run_ai_cleanup(
                    content,
                    backend,
                    context_chars=max(0, _int_from_meta(settings["context_chars"], 0)),
                )
        except Exception as exc:
            self._show_error(f"AI cleanup failed: {exc}")
            return
        if result.new_text == content:
            wx.MessageBox("AI cleanup produced no changes.", "AI Cleanup")
            return
        with AIDiffDialog(self, result.diff, result.block_count) as dialog:
            if dialog.ShowModal() != wx.ID_OK:
                return
        self.last_ai_undo_text = content
        self.undo_ai_btn.Enable(True)
        self.editor.SetValue(result.new_text)
        self.dirty = True
        self.save_current_node()

    def on_undo_ai(self, event: wx.CommandEvent) -> None:
        if self.last_ai_undo_text is None:
            return
        self.editor.SetValue(self.last_ai_undo_text)
        self.last_ai_undo_text = None
        self.undo_ai_btn.Enable(False)
        self.dirty = True
        self.save_current_node()

    def on_import_folder(self, event: wx.CommandEvent) -> None:
        with wx.DirDialog(self, "Choose Markdown folder to import") as dialog:
            if dialog.ShowModal() != wx.ID_OK:
                return
            ids = import_folder(self.repo, dialog.GetPath())
            self.selected_id = ids[0] if ids else None
            self.refresh_tree()
            self.load_selected_node()
            self._persist_selected_node()
            self._persist_expanded_nodes()
            self._persist_tree_top_visible_id()

    def on_export_selected(self, event: wx.CommandEvent) -> None:
        if self.selected_id is None:
            self._show_error("Select a node before exporting.")
            return
        self._export(self.selected_id)

    def on_export_all(self, event: wx.CommandEvent) -> None:
        self._export(None)

    def _export(self, root_id: int | None) -> None:
        with wx.DirDialog(self, "Choose export folder", style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dialog:
            if dialog.ShowModal() != wx.ID_OK:
                return
            written = export_tree(self.repo, dialog.GetPath(), root_id=root_id)
            wx.MessageBox(f"Exported {len(written)} Markdown files.", "Export Complete")

    def on_new_library(self, event: wx.CommandEvent) -> None:
        with wx.FileDialog(
            self,
            "Create SQLite library",
            wildcard="SQLite libraries (*.sqlite3)|*.sqlite3|All files (*.*)|*.*",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dialog:
            if dialog.ShowModal() != wx.ID_OK:
                return
            self._switch_library(Path(dialog.GetPath()))

    def on_open_library(self, event: wx.CommandEvent) -> None:
        with wx.FileDialog(
            self,
            "Open SQLite library",
            wildcard="SQLite libraries (*.sqlite3)|*.sqlite3|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dialog:
            if dialog.ShowModal() != wx.ID_OK:
                return
            self._switch_library(Path(dialog.GetPath()))

    def _switch_library(self, path: Path) -> None:
        self._persist_current_positions()
        self._persist_tree_top_visible_id()
        self.save_current_node()
        self.repo.close()
        self.repo = Repository(path)
        self.db_path = path
        self.selected_id = self._load_persisted_selected_id()
        self.theme = self._load_theme()
        self.ai_backend_type = self._load_ai_backend_type()
        self.llm_model_path = self._load_ai_model_path()
        if isinstance(self.ai_backend, LlamaCppBackend):
            wx.MessageBox(
                "A local llama.cpp model is already loaded. Restart the app before loading another library/model "
                "to avoid CUDA reload errors.",
                "Markdown Editor",
                wx.OK | wx.ICON_INFORMATION,
            )
        else:
            self.ai_backend = None
            self.local_backend_signature = None
        self.last_ai_undo_text = None
        self.undo_ai_btn.Enable(False)
        self.night_mode_item.Check(self.theme == "dark")
        self.apply_theme()
        self.SetStatusText(str(path))
        self.refresh_tree()
        self.load_selected_node()

    def on_close(self, event: wx.CloseEvent) -> None:
        try:
            self._persist_current_positions()
            self.save_current_node()
            self._persist_selected_node()
            self._persist_expanded_nodes()
            self._persist_tree_top_visible_id()
        except RepositoryError as exc:
            self._show_error(str(exc))
            event.Veto()
            return
        event.Skip()

    def _update_button_state(self) -> None:
        has_selection = self.selected_id is not None
        self.add_child_btn.Enable(has_selection)
        self.rename_btn.Enable(has_selection)
        self.delete_btn.Enable(has_selection)
        self.export_selected_item.Enable(has_selection)

    def _show_error(self, message: str) -> None:
        wx.MessageBox(message, "Markdown Editor", wx.OK | wx.ICON_ERROR)

    def apply_theme(self) -> None:
        colors = THEMES[self.theme]
        self.SetBackgroundColour(colors["background"])
        for panel in (self.tree_panel, self.edit_panel, self.view_panel, self.content_notebook, self.splitter):
            panel.SetBackgroundColour(colors["panel"])
            panel.SetForegroundColour(colors["foreground"])
        self.tree.SetBackgroundColour(colors["tree_background"])
        self.tree.SetForegroundColour(colors["tree_foreground"])
        self.title_ctrl.SetBackgroundColour(colors["editor_background"])
        self.title_ctrl.SetForegroundColour(colors["editor_foreground"])
        self.editor.SetBackgroundColour(colors["editor_background"])
        self.editor.SetForegroundColour(colors["editor_foreground"])
        self._apply_theme_to_children(self.tree_panel, colors)
        self._apply_theme_to_children(self.edit_panel, colors)
        self.Refresh()

    def _apply_theme_to_children(self, window: wx.Window, colors: dict[str, str]) -> None:
        for child in window.GetChildren():
            if isinstance(child, (wx.Button, wx.TextCtrl, wx.TreeCtrl)):
                continue
            child.SetBackgroundColour(colors["panel"])
            child.SetForegroundColour(colors["foreground"])
            self._apply_theme_to_children(child, colors)

    def _rerender_preview(self) -> None:
        if self.selected_id is None:
            self.preview.SetPage(render_markdown("", self.theme), "")
            return
        self.pending_view_scroll_y = self._load_view_scroll_y(self.selected_id)
        self.preview.SetPage(render_markdown(self.repo.get_node(self.selected_id).markdown_content, self.theme), "")

    def _load_theme(self) -> str:
        theme = self.repo.get_meta(THEME_META_KEY, "light")
        return theme if theme in THEMES else "light"

    def _load_ai_model_path(self) -> Path | None:
        raw = self.repo.get_meta(AI_MODEL_PATH_META_KEY)
        return Path(raw) if raw else None

    def _load_ai_backend_type(self) -> str:
        backend = self.repo.get_meta(AI_BACKEND_META_KEY, AI_BACKEND_LOCAL)
        return backend if backend in {AI_BACKEND_LOCAL, AI_BACKEND_OPENAI} else AI_BACKEND_LOCAL

    def _load_ai_settings(self) -> dict[str, str]:
        return {
            "backend": self._load_ai_backend_type(),
            "local_model_path": str(self._load_ai_model_path() or ""),
            "local_max_tokens": self.repo.get_meta(AI_LOCAL_MAX_TOKENS_META_KEY, "350") or "350",
            "local_temperature": self.repo.get_meta(AI_LOCAL_TEMPERATURE_META_KEY, "0.1") or "0.1",
            "local_repeat_penalty": self.repo.get_meta(AI_LOCAL_REPEAT_PENALTY_META_KEY, "1.15") or "1.15",
            "local_top_k": self.repo.get_meta(AI_LOCAL_TOP_K_META_KEY, "40") or "40",
            "local_top_p": self.repo.get_meta(AI_LOCAL_TOP_P_META_KEY, "0.95") or "0.95",
            "local_min_p": self.repo.get_meta(AI_LOCAL_MIN_P_META_KEY, "0.05") or "0.05",
            "local_frequency_penalty": self.repo.get_meta(AI_LOCAL_FREQUENCY_PENALTY_META_KEY, "0.2") or "0.2",
            "local_presence_penalty": self.repo.get_meta(AI_LOCAL_PRESENCE_PENALTY_META_KEY, "0.0") or "0.0",
            "local_seed": self.repo.get_meta(AI_LOCAL_SEED_META_KEY, "42") or "42",
            "local_n_ctx": self.repo.get_meta(AI_LOCAL_CONTEXT_META_KEY, "4096") or "4096",
            "local_n_threads": self.repo.get_meta(AI_LOCAL_THREADS_META_KEY, "") or "",
            "local_n_gpu_layers": self.repo.get_meta(AI_LOCAL_GPU_LAYERS_META_KEY, "-1") or "-1",
            "local_thinking_mode": self.repo.get_meta(AI_LOCAL_THINKING_MODE_META_KEY, AI_THINKING_DISABLED)
            or AI_THINKING_DISABLED,
            "local_n_batch": self.repo.get_meta(AI_LOCAL_BATCH_META_KEY, "128") or "128",
            "local_n_ubatch": self.repo.get_meta(AI_LOCAL_UBATCH_META_KEY, "128") or "128",
            "local_offload_kqv": self.repo.get_meta(AI_LOCAL_OFFLOAD_KQV_META_KEY, "true") or "true",
            "local_flash_attn": self.repo.get_meta(AI_LOCAL_FLASH_ATTN_META_KEY, "false") or "false",
            "context_chars": self.repo.get_meta(AI_CONTEXT_CHARS_META_KEY, "0") or "0",
            "openai_base_url": self.repo.get_meta(AI_OPENAI_BASE_URL_META_KEY, "") or "",
            "openai_api_key": self.repo.get_meta(AI_OPENAI_API_KEY_META_KEY, "") or "",
            "openai_model": self.repo.get_meta(AI_OPENAI_MODEL_META_KEY, "") or "",
            "openai_max_tokens": self.repo.get_meta(AI_OPENAI_MAX_TOKENS_META_KEY, "700") or "700",
            "openai_temperature": self.repo.get_meta(AI_OPENAI_TEMPERATURE_META_KEY, "0.2") or "0.2",
            "openai_top_p": self.repo.get_meta(AI_OPENAI_TOP_P_META_KEY, "1.0") or "1.0",
            "openai_frequency_penalty": self.repo.get_meta(AI_OPENAI_FREQUENCY_PENALTY_META_KEY, "0.0") or "0.0",
            "openai_presence_penalty": self.repo.get_meta(AI_OPENAI_PRESENCE_PENALTY_META_KEY, "0.0") or "0.0",
            "openai_timeout_seconds": self.repo.get_meta(AI_OPENAI_TIMEOUT_META_KEY, "120") or "120",
        }

    def _save_ai_settings(self, settings: dict[str, str]) -> None:
        self.repo.set_meta(AI_BACKEND_META_KEY, settings["backend"])
        self.repo.set_meta(AI_MODEL_PATH_META_KEY, settings["local_model_path"])
        self.repo.set_meta(AI_LOCAL_MAX_TOKENS_META_KEY, settings["local_max_tokens"])
        self.repo.set_meta(AI_LOCAL_TEMPERATURE_META_KEY, settings["local_temperature"])
        self.repo.set_meta(AI_LOCAL_REPEAT_PENALTY_META_KEY, settings["local_repeat_penalty"])
        self.repo.set_meta(AI_LOCAL_TOP_K_META_KEY, settings["local_top_k"])
        self.repo.set_meta(AI_LOCAL_TOP_P_META_KEY, settings["local_top_p"])
        self.repo.set_meta(AI_LOCAL_MIN_P_META_KEY, settings["local_min_p"])
        self.repo.set_meta(AI_LOCAL_FREQUENCY_PENALTY_META_KEY, settings["local_frequency_penalty"])
        self.repo.set_meta(AI_LOCAL_PRESENCE_PENALTY_META_KEY, settings["local_presence_penalty"])
        self.repo.set_meta(AI_LOCAL_SEED_META_KEY, settings["local_seed"])
        self.repo.set_meta(AI_LOCAL_CONTEXT_META_KEY, settings["local_n_ctx"])
        self.repo.set_meta(AI_LOCAL_THREADS_META_KEY, settings["local_n_threads"])
        self.repo.set_meta(AI_LOCAL_GPU_LAYERS_META_KEY, settings["local_n_gpu_layers"])
        self.repo.set_meta(AI_LOCAL_THINKING_MODE_META_KEY, settings["local_thinking_mode"])
        self.repo.set_meta(AI_LOCAL_BATCH_META_KEY, settings["local_n_batch"])
        self.repo.set_meta(AI_LOCAL_UBATCH_META_KEY, settings["local_n_ubatch"])
        self.repo.set_meta(AI_LOCAL_OFFLOAD_KQV_META_KEY, settings["local_offload_kqv"])
        self.repo.set_meta(AI_LOCAL_FLASH_ATTN_META_KEY, settings["local_flash_attn"])
        self.repo.set_meta(AI_CONTEXT_CHARS_META_KEY, settings["context_chars"])
        self.repo.set_meta(AI_OPENAI_BASE_URL_META_KEY, settings["openai_base_url"])
        self.repo.set_meta(AI_OPENAI_API_KEY_META_KEY, settings["openai_api_key"])
        self.repo.set_meta(AI_OPENAI_MODEL_META_KEY, settings["openai_model"])
        self.repo.set_meta(AI_OPENAI_MAX_TOKENS_META_KEY, settings["openai_max_tokens"])
        self.repo.set_meta(AI_OPENAI_TEMPERATURE_META_KEY, settings["openai_temperature"])
        self.repo.set_meta(AI_OPENAI_TOP_P_META_KEY, settings["openai_top_p"])
        self.repo.set_meta(AI_OPENAI_FREQUENCY_PENALTY_META_KEY, settings["openai_frequency_penalty"])
        self.repo.set_meta(AI_OPENAI_PRESENCE_PENALTY_META_KEY, settings["openai_presence_penalty"])
        self.repo.set_meta(AI_OPENAI_TIMEOUT_META_KEY, settings["openai_timeout_seconds"])

    def _prompt_ai_model_path(self) -> bool:
        with wx.FileDialog(
            self,
            "Select llama.cpp GGUF model",
            wildcard="GGUF models (*.gguf)|*.gguf|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dialog:
            if dialog.ShowModal() != wx.ID_OK:
                return False
            self.llm_model_path = Path(dialog.GetPath())
            self.ai_backend = None
            self.repo.set_meta(AI_MODEL_PATH_META_KEY, str(self.llm_model_path))
            self.SetStatusText(f"AI model: {self.llm_model_path}")
            return True

    def _get_ai_backend(self) -> object | None:
        if self.ai_backend_type == AI_BACKEND_LOCAL and isinstance(self.ai_backend, LlamaCppBackend):
            return self.ai_backend
        if self.ai_backend_type == AI_BACKEND_OPENAI:
            if isinstance(self.ai_backend, OpenAICompatBackend):
                return self.ai_backend
            return self._get_openai_compat_backend()
        return self._get_local_llm_backend()

    def _get_local_llm_backend(self) -> LlamaCppBackend | None:
        if self.llm_model_path is None or not self.llm_model_path.exists():
            if not self._prompt_ai_model_path():
                return None
        if self.llm_model_path is None:
            return None
        try:
            with wx.BusyInfo("Loading AI model..."):
                settings = self._load_ai_settings()
                signature = self._local_backend_signature(settings)
                self.ai_backend = LlamaCppBackend(
                    self.llm_model_path,
                    n_ctx=_int_from_meta(settings["local_n_ctx"], 4096),
                    max_tokens=_int_from_meta(settings["local_max_tokens"], 350),
                    temperature=_float_from_meta(settings["local_temperature"], 0.1),
                    repeat_penalty=_float_from_meta(settings["local_repeat_penalty"], 1.15),
                    top_k=_int_from_meta(settings["local_top_k"], 40),
                    top_p=_float_from_meta(settings["local_top_p"], 0.95),
                    min_p=_float_from_meta(settings["local_min_p"], 0.05),
                    frequency_penalty=_float_from_meta(settings["local_frequency_penalty"], 0.2),
                    presence_penalty=_float_from_meta(settings["local_presence_penalty"], 0.0),
                    seed=_optional_int_from_meta(settings["local_seed"]),
                    n_threads=_optional_int_from_meta(settings["local_n_threads"]),
                    n_gpu_layers=_int_from_meta(settings["local_n_gpu_layers"], -1),
                    disable_thinking=settings["local_thinking_mode"] != AI_THINKING_ALLOWED,
                    n_batch=_int_from_meta(settings["local_n_batch"], 128),
                    n_ubatch=_int_from_meta(settings["local_n_ubatch"], 128),
                    offload_kqv=_bool_from_meta(settings["local_offload_kqv"], True),
                    flash_attn=_bool_from_meta(settings["local_flash_attn"], False),
                )
                self.local_backend_signature = signature
        except RuntimeError as exc:
            self._show_error(str(exc))
            return None
        except Exception as exc:
            self._show_error(f"Could not load AI model: {exc}")
            return None
        return self.ai_backend

    @staticmethod
    def _local_backend_signature(settings: dict[str, str]) -> tuple[str, ...]:
        return (
            settings.get("local_model_path", ""),
            settings.get("local_n_ctx", ""),
            settings.get("local_max_tokens", ""),
            settings.get("local_temperature", ""),
            settings.get("local_repeat_penalty", ""),
            settings.get("local_top_k", ""),
            settings.get("local_top_p", ""),
            settings.get("local_min_p", ""),
            settings.get("local_frequency_penalty", ""),
            settings.get("local_presence_penalty", ""),
            settings.get("local_seed", ""),
            settings.get("local_n_threads", ""),
            settings.get("local_n_gpu_layers", ""),
            settings.get("local_thinking_mode", ""),
            settings.get("local_n_batch", ""),
            settings.get("local_n_ubatch", ""),
            settings.get("local_offload_kqv", ""),
            settings.get("local_flash_attn", ""),
        )

    def _get_openai_compat_backend(self) -> OpenAICompatBackend | None:
        settings = self._load_ai_settings()
        base_url = settings["openai_base_url"].strip()
        model = settings["openai_model"].strip()
        if not base_url or not model:
            self._show_error("Configure OpenAI-compatible base URL and model in AI -> Settings first.")
            return None
        self.ai_backend = OpenAICompatBackend(
            base_url=base_url,
            api_key=settings["openai_api_key"],
            model=model,
            max_tokens=_int_from_meta(settings["openai_max_tokens"], 700),
            temperature=_float_from_meta(settings["openai_temperature"], 0.2),
            top_p=_float_from_meta(settings["openai_top_p"], 1.0),
            frequency_penalty=_float_from_meta(settings["openai_frequency_penalty"], 0.0),
            presence_penalty=_float_from_meta(settings["openai_presence_penalty"], 0.0),
            timeout_seconds=_int_from_meta(settings["openai_timeout_seconds"], 120),
        )
        return self.ai_backend

    def _load_persisted_selected_id(self) -> int | None:
        raw = self.repo.get_meta(SELECTED_NODE_META_KEY)
        if not raw:
            return None
        try:
            node_id = int(raw)
            self.repo.get_node(node_id)
            return node_id
        except (ValueError, NodeNotFoundError):
            return None

    def _persist_selected_node(self) -> None:
        self.repo.set_meta(SELECTED_NODE_META_KEY, "" if self.selected_id is None else str(self.selected_id))

    def _load_persisted_expanded_ids(self) -> set[int]:
        raw = self.repo.get_meta(EXPANDED_NODES_META_KEY, "[]")
        try:
            values = json.loads(raw or "[]")
        except json.JSONDecodeError:
            return set()
        if not isinstance(values, list):
            return set()
        valid: set[int] = set()
        for value in values:
            if not isinstance(value, int):
                continue
            try:
                self.repo.get_node(value)
            except NodeNotFoundError:
                continue
            valid.add(value)
        return valid

    def _persist_expanded_nodes(self) -> None:
        root = self.tree.GetRootItem()
        expanded = sorted(self._collect_expanded_node_ids(root)) if root.IsOk() else []
        self.repo.set_meta(EXPANDED_NODES_META_KEY, json.dumps(expanded))

    def _collect_expanded_node_ids(self, item: wx.TreeItemId) -> set[int]:
        expanded: set[int] = set()
        node_id = self.tree.GetItemData(item)
        if node_id is not None and self.tree.IsExpanded(item):
            expanded.add(node_id)
        child, cookie = self.tree.GetFirstChild(item)
        while child.IsOk():
            expanded.update(self._collect_expanded_node_ids(child))
            child, cookie = self.tree.GetNextChild(item, cookie)
        return expanded

    def _persist_tree_top_visible_id(self) -> None:
        item = self.tree.GetFirstVisibleItem()
        if not item.IsOk():
            self.repo.set_meta(TREE_TOP_NODE_META_KEY, "")
            return
        node_id = self.tree.GetItemData(item)
        self.repo.set_meta(TREE_TOP_NODE_META_KEY, "" if node_id is None else str(node_id))

    def _restore_tree_top_visible_id(self) -> None:
        raw = self.repo.get_meta(TREE_TOP_NODE_META_KEY)
        if not raw:
            return
        try:
            node_id = int(raw)
        except ValueError:
            return
        item = self._find_tree_item_by_node_id(self.tree.GetRootItem(), node_id)
        if item is not None:
            self.tree.EnsureVisible(item)

    def _find_tree_item_by_node_id(self, item: wx.TreeItemId, node_id: int) -> wx.TreeItemId | None:
        if not item.IsOk():
            return None
        if self.tree.GetItemData(item) == node_id:
            return item
        child, cookie = self.tree.GetFirstChild(item)
        while child.IsOk():
            found = self._find_tree_item_by_node_id(child, node_id)
            if found is not None:
                return found
            child, cookie = self.tree.GetNextChild(item, cookie)
        return None

    def _persist_current_positions(self, page_label: str | None = None) -> None:
        if self.selected_id is None:
            return
        label = page_label or self._current_content_page_label()
        if label == "Edit":
            self._persist_editor_position(self.selected_id)
        elif label == "View":
            self._persist_view_position(self.selected_id)

    def _current_content_page_label(self) -> str:
        return self.content_notebook.GetPageText(self.content_notebook.GetSelection())

    def _load_json_meta_map(self, key: str) -> dict[str, object]:
        raw = self.repo.get_meta(key, "{}")
        try:
            value = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def _store_json_meta_map_value(self, key: str, node_id: int, value: dict[str, int]) -> None:
        values = self._load_json_meta_map(key)
        values[str(node_id)] = value
        self.repo.set_meta(key, json.dumps(values, sort_keys=True))

    def _persist_editor_position(self, node_id: int) -> None:
        self._store_json_meta_map_value(
            EDITOR_POSITIONS_META_KEY,
            node_id,
            {
                "insertion_point": self.editor.GetInsertionPoint(),
                "scroll_y": self._get_window_scroll_pos(self.editor),
            },
        )

    def _restore_editor_position(self, node_id: int) -> None:
        values = self._load_json_meta_map(EDITOR_POSITIONS_META_KEY).get(str(node_id), {})
        if not isinstance(values, dict):
            return
        insertion_point = _int_from_meta(values.get("insertion_point"), 0)
        insertion_point = max(0, min(insertion_point, len(self.editor.GetValue())))
        self.editor.SetInsertionPoint(insertion_point)
        self.editor.ShowPosition(insertion_point)
        scroll_y = _int_from_meta(values.get("scroll_y"), 0)
        if scroll_y > 0:
            self._set_window_scroll_pos(self.editor, scroll_y)

    def _persist_view_position(self, node_id: int) -> None:
        self._store_json_meta_map_value(
            VIEW_POSITIONS_META_KEY,
            node_id,
            {"scroll_y": self._get_view_scroll_y()},
        )

    def _load_view_scroll_y(self, node_id: int) -> int:
        values = self._load_json_meta_map(VIEW_POSITIONS_META_KEY).get(str(node_id), {})
        if not isinstance(values, dict):
            return 0
        return _int_from_meta(values.get("scroll_y"), 0)

    def _get_view_scroll_y(self) -> int:
        try:
            result = self.preview.RunScript("String(Math.round(window.scrollY || window.pageYOffset || 0));")
        except Exception:
            return 0
        if isinstance(result, tuple):
            if len(result) >= 2 and result[0]:
                return _int_from_meta(result[1], 0)
            return 0
        return _int_from_meta(result, 0)

    def _set_view_scroll_y(self, scroll_y: int) -> None:
        try:
            self.preview.RunScript(f"window.scrollTo(0, {max(0, int(scroll_y))});")
        except Exception:
            pass

    @staticmethod
    def _get_window_scroll_pos(window: wx.Window) -> int:
        try:
            return int(window.GetScrollPos(wx.VERTICAL))
        except Exception:
            return 0

    @staticmethod
    def _set_window_scroll_pos(window: wx.Window, position: int) -> None:
        try:
            window.SetScrollPos(wx.VERTICAL, max(0, int(position)), True)
        except Exception:
            pass

    @staticmethod
    def _set_splitter_sash_size(splitter: wx.SplitterWindow, size: int) -> None:
        try:
            splitter.SashSize = size
        except Exception:
            pass

    @staticmethod
    def _disable_smart_text_substitutions(text_ctrl: wx.TextCtrl) -> None:
        if wx.Platform != "__WXMAC__":
            return
        disable_all = getattr(text_ctrl, "OSXDisableAllSmartSubstitutions", None)
        if callable(disable_all):
            try:
                disable_all()
            except (NotImplementedError, RuntimeError):
                pass
            return
        for method_name in (
            "OSXEnableAutomaticDashSubstitution",
            "OSXEnableAutomaticQuoteSubstitution",
            "OSXEnableNewLineReplacement",
        ):
            method = getattr(text_ctrl, method_name, None)
            if callable(method):
                try:
                    method(False)
                except (NotImplementedError, RuntimeError):
                    pass


def _int_from_meta(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_from_meta(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_int_from_meta(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_from_meta(value: object, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


class AISettingsDialog(wx.Dialog):
    def __init__(self, parent: wx.Window, settings: dict[str, str]):
        super().__init__(parent, title="AI Settings", size=(760, 620))
        self.settings = settings
        self.local_ai_available = is_llama_cpp_available()
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.backend_choice = wx.RadioBox(
            self,
            label="Backend",
            choices=["Local llama.cpp", "OpenAI-compatible API"],
            majorDimension=1,
            style=wx.RA_SPECIFY_ROWS,
        )
        self.backend_choice.SetSelection(1 if settings["backend"] == AI_BACKEND_OPENAI or not self.local_ai_available else 0)
        if not self.local_ai_available:
            self.backend_choice.EnableItem(0, False)
        sizer.Add(self.backend_choice, 0, wx.EXPAND | wx.ALL, 8)

        context_sizer = wx.BoxSizer(wx.HORIZONTAL)
        context_sizer.Add(wx.StaticText(self, label="Context around block"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        self.context_chars = wx.TextCtrl(self, value=settings["context_chars"], size=(90, -1))
        context_sizer.Add(self.context_chars, 0, wx.RIGHT, 8)
        context_sizer.Add(
            wx.StaticText(self, label="characters before/after marked block"),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        sizer.Add(context_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        scrolled = wx.ScrolledWindow(self, style=wx.VSCROLL)
        scrolled.SetScrollRate(0, 12)
        scrolled_sizer = wx.BoxSizer(wx.VERTICAL)
        local_box = wx.StaticBoxSizer(wx.VERTICAL, scrolled, "Local llama.cpp")
        self.local_unavailable_note = wx.StaticText(
            scrolled,
            label="llama-cpp-python is not installed. Local model settings are unavailable.",
        )
        local_box.Add(self.local_unavailable_note, 0, wx.EXPAND | wx.ALL, 8)
        self.local_unavailable_note.Show(not self.local_ai_available)
        local_grid = wx.FlexGridSizer(rows=0, cols=3, vgap=8, hgap=8)
        local_grid.AddGrowableCol(1, 1)

        self.local_model_path = wx.TextCtrl(scrolled, value=settings["local_model_path"])
        self.browse_model_btn = wx.Button(scrolled, label="Browse...")
        local_grid.Add(wx.StaticText(scrolled, label="GGUF model"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_model_path, 1, wx.EXPAND)
        local_grid.Add(self.browse_model_btn, 0)

        self.local_max_tokens = wx.TextCtrl(scrolled, value=settings["local_max_tokens"])
        local_grid.Add(wx.StaticText(scrolled, label="Max tokens"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_max_tokens, 1, wx.EXPAND)
        local_grid.AddSpacer(1)

        self.local_temperature = wx.TextCtrl(scrolled, value=settings["local_temperature"])
        local_grid.Add(wx.StaticText(scrolled, label="Temperature"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_temperature, 1, wx.EXPAND)
        local_grid.AddSpacer(1)

        self.local_repeat_penalty = wx.TextCtrl(scrolled, value=settings["local_repeat_penalty"])
        local_grid.Add(wx.StaticText(scrolled, label="Repeat penalty"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_repeat_penalty, 1, wx.EXPAND)
        local_grid.AddSpacer(1)

        self.local_top_k = wx.TextCtrl(scrolled, value=settings["local_top_k"])
        local_grid.Add(wx.StaticText(scrolled, label="Top K"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_top_k, 1, wx.EXPAND)
        local_grid.AddSpacer(1)

        self.local_top_p = wx.TextCtrl(scrolled, value=settings["local_top_p"])
        local_grid.Add(wx.StaticText(scrolled, label="Top P"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_top_p, 1, wx.EXPAND)
        local_grid.AddSpacer(1)

        self.local_min_p = wx.TextCtrl(scrolled, value=settings["local_min_p"])
        local_grid.Add(wx.StaticText(scrolled, label="Min P"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_min_p, 1, wx.EXPAND)
        local_grid.AddSpacer(1)

        self.local_frequency_penalty = wx.TextCtrl(scrolled, value=settings["local_frequency_penalty"])
        local_grid.Add(wx.StaticText(scrolled, label="Frequency penalty"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_frequency_penalty, 1, wx.EXPAND)
        local_grid.AddSpacer(1)

        self.local_presence_penalty = wx.TextCtrl(scrolled, value=settings["local_presence_penalty"])
        local_grid.Add(wx.StaticText(scrolled, label="Presence penalty"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_presence_penalty, 1, wx.EXPAND)
        local_grid.AddSpacer(1)

        self.local_seed = wx.TextCtrl(scrolled, value=settings["local_seed"])
        local_grid.Add(wx.StaticText(scrolled, label="Seed"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_seed, 1, wx.EXPAND)
        local_grid.AddSpacer(1)

        self.local_n_ctx = wx.TextCtrl(scrolled, value=settings["local_n_ctx"])
        local_grid.Add(wx.StaticText(scrolled, label="Context"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_n_ctx, 1, wx.EXPAND)
        local_grid.AddSpacer(1)

        self.local_n_threads = wx.TextCtrl(scrolled, value=settings["local_n_threads"])
        local_grid.Add(wx.StaticText(scrolled, label="Threads"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_n_threads, 1, wx.EXPAND)
        local_grid.AddSpacer(1)

        self.local_n_gpu_layers = wx.TextCtrl(scrolled, value=settings["local_n_gpu_layers"])
        local_grid.Add(wx.StaticText(scrolled, label="GPU layers"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_n_gpu_layers, 1, wx.EXPAND)
        local_grid.AddSpacer(1)

        self.local_thinking_mode = wx.Choice(scrolled, choices=["Disable thinking", "Allow thinking"])
        self.local_thinking_mode.SetSelection(1 if settings["local_thinking_mode"] == AI_THINKING_ALLOWED else 0)
        local_grid.Add(wx.StaticText(scrolled, label="Thinking mode"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_thinking_mode, 1, wx.EXPAND)
        local_grid.AddSpacer(1)

        self.local_n_batch = wx.TextCtrl(scrolled, value=settings["local_n_batch"])
        local_grid.Add(wx.StaticText(scrolled, label="Batch"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_n_batch, 1, wx.EXPAND)
        local_grid.AddSpacer(1)

        self.local_n_ubatch = wx.TextCtrl(scrolled, value=settings["local_n_ubatch"])
        local_grid.Add(wx.StaticText(scrolled, label="Micro batch"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_n_ubatch, 1, wx.EXPAND)
        local_grid.AddSpacer(1)

        self.local_offload_kqv = wx.CheckBox(scrolled, label="Offload K/Q/V to GPU")
        self.local_offload_kqv.SetValue(_bool_from_meta(settings["local_offload_kqv"], True))
        local_grid.Add(wx.StaticText(scrolled, label="K/Q/V offload"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_offload_kqv, 1, wx.EXPAND)
        local_grid.AddSpacer(1)

        self.local_flash_attn = wx.CheckBox(scrolled, label="Use flash attention")
        self.local_flash_attn.SetValue(_bool_from_meta(settings["local_flash_attn"], False))
        local_grid.Add(wx.StaticText(scrolled, label="Flash attention"), 0, wx.ALIGN_CENTER_VERTICAL)
        local_grid.Add(self.local_flash_attn, 1, wx.EXPAND)
        local_grid.AddSpacer(1)
        local_box.Add(local_grid, 0, wx.EXPAND | wx.ALL, 8)

        api_box = wx.StaticBoxSizer(wx.VERTICAL, scrolled, "OpenAI-compatible API")
        api_grid = wx.FlexGridSizer(rows=0, cols=3, vgap=8, hgap=8)
        api_grid.AddGrowableCol(1, 1)
        self.openai_base_url = wx.TextCtrl(scrolled, value=settings["openai_base_url"])
        api_grid.Add(wx.StaticText(scrolled, label="Base URL"), 0, wx.ALIGN_CENTER_VERTICAL)
        api_grid.Add(self.openai_base_url, 1, wx.EXPAND)
        api_grid.AddSpacer(1)

        self.openai_api_key = wx.TextCtrl(scrolled, value=settings["openai_api_key"], style=wx.TE_PASSWORD)
        api_grid.Add(wx.StaticText(scrolled, label="API key"), 0, wx.ALIGN_CENTER_VERTICAL)
        api_grid.Add(self.openai_api_key, 1, wx.EXPAND)
        api_grid.AddSpacer(1)

        self.openai_model = wx.TextCtrl(scrolled, value=settings["openai_model"])
        api_grid.Add(wx.StaticText(scrolled, label="Model"), 0, wx.ALIGN_CENTER_VERTICAL)
        api_grid.Add(self.openai_model, 1, wx.EXPAND)
        api_grid.AddSpacer(1)

        self.openai_max_tokens = wx.TextCtrl(scrolled, value=settings["openai_max_tokens"])
        api_grid.Add(wx.StaticText(scrolled, label="Max tokens"), 0, wx.ALIGN_CENTER_VERTICAL)
        api_grid.Add(self.openai_max_tokens, 1, wx.EXPAND)
        api_grid.AddSpacer(1)

        self.openai_temperature = wx.TextCtrl(scrolled, value=settings["openai_temperature"])
        api_grid.Add(wx.StaticText(scrolled, label="Temperature"), 0, wx.ALIGN_CENTER_VERTICAL)
        api_grid.Add(self.openai_temperature, 1, wx.EXPAND)
        api_grid.AddSpacer(1)

        self.openai_top_p = wx.TextCtrl(scrolled, value=settings["openai_top_p"])
        api_grid.Add(wx.StaticText(scrolled, label="Top P"), 0, wx.ALIGN_CENTER_VERTICAL)
        api_grid.Add(self.openai_top_p, 1, wx.EXPAND)
        api_grid.AddSpacer(1)

        self.openai_frequency_penalty = wx.TextCtrl(scrolled, value=settings["openai_frequency_penalty"])
        api_grid.Add(wx.StaticText(scrolled, label="Frequency penalty"), 0, wx.ALIGN_CENTER_VERTICAL)
        api_grid.Add(self.openai_frequency_penalty, 1, wx.EXPAND)
        api_grid.AddSpacer(1)

        self.openai_presence_penalty = wx.TextCtrl(scrolled, value=settings["openai_presence_penalty"])
        api_grid.Add(wx.StaticText(scrolled, label="Presence penalty"), 0, wx.ALIGN_CENTER_VERTICAL)
        api_grid.Add(self.openai_presence_penalty, 1, wx.EXPAND)
        api_grid.AddSpacer(1)

        self.openai_timeout_seconds = wx.TextCtrl(scrolled, value=settings["openai_timeout_seconds"])
        api_grid.Add(wx.StaticText(scrolled, label="Timeout seconds"), 0, wx.ALIGN_CENTER_VERTICAL)
        api_grid.Add(self.openai_timeout_seconds, 1, wx.EXPAND)
        api_grid.AddSpacer(1)
        api_box.Add(api_grid, 0, wx.EXPAND | wx.ALL, 8)

        scrolled_sizer.Add(local_box, 0, wx.EXPAND | wx.ALL, 8)
        scrolled_sizer.Add(api_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        scrolled.SetSizer(scrolled_sizer)
        sizer.Add(scrolled, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        note = wx.StaticText(self, label="API key is stored in this SQLite library as plain text.")
        sizer.Add(note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        button_sizer = wx.StdDialogButtonSizer()
        ok_btn = wx.Button(self, wx.ID_OK, "Save")
        cancel_btn = wx.Button(self, wx.ID_CANCEL, "Cancel")
        button_sizer.AddButton(ok_btn)
        button_sizer.AddButton(cancel_btn)
        button_sizer.Realize()
        sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 8)

        self.SetSizer(sizer)
        self.browse_model_btn.Bind(wx.EVT_BUTTON, self.on_browse_model)
        self._set_sizer_enabled(local_grid, self.local_ai_available)
        local_box.GetStaticBox().Enable(self.local_ai_available)
        scrolled_sizer.Layout()

    def on_browse_model(self, event: wx.CommandEvent) -> None:
        with wx.FileDialog(
            self,
            "Select llama.cpp GGUF model",
            wildcard="GGUF models (*.gguf)|*.gguf|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dialog:
            if dialog.ShowModal() != wx.ID_OK:
                return
            self.local_model_path.SetValue(dialog.GetPath())

    def get_settings(self) -> dict[str, str]:
        return {
            "backend": AI_BACKEND_LOCAL
            if self.local_ai_available and self.backend_choice.GetSelection() == 0
            else AI_BACKEND_OPENAI,
            "local_model_path": self.local_model_path.GetValue().strip(),
            "local_max_tokens": self.local_max_tokens.GetValue().strip() or "350",
            "local_temperature": self.local_temperature.GetValue().strip() or "0.1",
            "local_repeat_penalty": self.local_repeat_penalty.GetValue().strip() or "1.15",
            "local_top_k": self.local_top_k.GetValue().strip() or "40",
            "local_top_p": self.local_top_p.GetValue().strip() or "0.95",
            "local_min_p": self.local_min_p.GetValue().strip() or "0.05",
            "local_frequency_penalty": self.local_frequency_penalty.GetValue().strip() or "0.2",
            "local_presence_penalty": self.local_presence_penalty.GetValue().strip() or "0.0",
            "local_seed": self.local_seed.GetValue().strip(),
            "local_n_ctx": self.local_n_ctx.GetValue().strip() or "4096",
            "local_n_threads": self.local_n_threads.GetValue().strip(),
            "local_n_gpu_layers": self.local_n_gpu_layers.GetValue().strip() or "-1",
            "local_thinking_mode": AI_THINKING_ALLOWED
            if self.local_thinking_mode.GetSelection() == 1
            else AI_THINKING_DISABLED,
            "local_n_batch": self.local_n_batch.GetValue().strip() or "128",
            "local_n_ubatch": self.local_n_ubatch.GetValue().strip() or "128",
            "local_offload_kqv": "true" if self.local_offload_kqv.GetValue() else "false",
            "local_flash_attn": "true" if self.local_flash_attn.GetValue() else "false",
            "context_chars": self.context_chars.GetValue().strip() or "0",
            "openai_base_url": self.openai_base_url.GetValue().strip(),
            "openai_api_key": self.openai_api_key.GetValue(),
            "openai_model": self.openai_model.GetValue().strip(),
            "openai_max_tokens": self.openai_max_tokens.GetValue().strip() or "700",
            "openai_temperature": self.openai_temperature.GetValue().strip() or "0.2",
            "openai_top_p": self.openai_top_p.GetValue().strip() or "1.0",
            "openai_frequency_penalty": self.openai_frequency_penalty.GetValue().strip() or "0.0",
            "openai_presence_penalty": self.openai_presence_penalty.GetValue().strip() or "0.0",
            "openai_timeout_seconds": self.openai_timeout_seconds.GetValue().strip() or "120",
        }

    def _set_sizer_enabled(self, sizer: wx.Sizer, enabled: bool) -> None:
        for item in sizer.GetChildren():
            window = item.GetWindow()
            child_sizer = item.GetSizer()
            if window is not None:
                window.Enable(enabled)
            if child_sizer is not None:
                self._set_sizer_enabled(child_sizer, enabled)


class AIDiffDialog(wx.Dialog):
    def __init__(self, parent: wx.Window, diff_text: str, block_count: int):
        super().__init__(parent, title="AI Cleanup Diff", size=(900, 650))
        sizer = wx.BoxSizer(wx.VERTICAL)
        label = wx.StaticText(self, label=f"AI proposed changes for {block_count} block(s).")
        self.diff_ctrl = wx.TextCtrl(
            self,
            value=diff_text,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2 | wx.HSCROLL,
        )
        font = wx.Font(
            10,
            wx.FONTFAMILY_TELETYPE,
            wx.FONTSTYLE_NORMAL,
            wx.FONTWEIGHT_NORMAL,
        )
        self.diff_ctrl.SetFont(font)
        button_sizer = wx.StdDialogButtonSizer()
        apply_btn = wx.Button(self, wx.ID_OK, "Apply")
        decline_btn = wx.Button(self, wx.ID_CANCEL, "Decline")
        button_sizer.AddButton(apply_btn)
        button_sizer.AddButton(decline_btn)
        button_sizer.Realize()

        sizer.Add(label, 0, wx.ALL, 8)
        sizer.Add(self.diff_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)
        sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 8)
        self.SetSizer(sizer)
