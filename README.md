# Portable Markdown Editor

A local, portable Markdown editor backed by SQLite. The GUI is built with wxPython and uses three tabs:

- Tree: organize Markdown nodes
- Edit: edit the selected node as plain Markdown
- View: preview the selected node as rendered HTML

## Run From Source

```powershell
python -m pip install -e ".[dev]"
python -m md_editor.app
```

The default library is created at `data/library.sqlite3` beside the project or packaged app.

Folder import creates a node for the selected folder, empty nodes for subfolders that contain Markdown files, and document nodes for `.md` / `.markdown` files. After import, this is just an editable node tree stored in SQLite.

## AI Cleanup

AI cleanup can use either a local `llama-cpp-python` GGUF model or an OpenAI-compatible HTTP API.

For local GGUF models, install the optional LLM extra:

```powershell
python -m pip install -e ".[dev,llm]"
```

Mark text in a document like this:

```markdown
<!-- AI: rewrite next paragraph to be clearer, preserve meaning -->
Text to rewrite.
<!-- AI: end -->
```

Use `AI -> Settings...` to choose:

- `Local llama.cpp`: select a `.gguf` model file.
- `OpenAI-compatible API`: enter base URL, API key, model name, max tokens, temperature, and timeout.

For local models, the settings dialog also exposes max tokens, temperature, top-k, top-p, min-p, frequency/presence penalties, repeat penalty, seed, context size, thread count, GPU layers, thinking mode, batch, micro batch, K/Q/V offload, and flash attention. Defaults are conservative for cleanup tasks: fixed seed, thinking disabled, low temperature, shorter output, smaller CUDA batches, and a repetition penalty.

The local seed defaults to `42`, making repeated cleanup runs on unchanged text deterministic. Leave it empty if you intentionally want varied outputs.

If CUDA crashes with larger models, first try `Batch=128`, `Micro batch=128`, reduce max tokens/context, then reduce GPU layers. Disabling K/Q/V offload can reduce GPU pressure at the cost of speed.

The OpenAI-compatible backend calls:

```text
{base_url}/chat/completions
```

The app shows a read-only diff before applying changes, and `Undo AI` restores the previous document text for the last accepted AI cleanup.

AI settings are stored in the current SQLite library. API keys are stored as plain text.

## Tests

```powershell
pytest
```

## Portable Build

```powershell
pyinstaller --noconfirm --windowed --name md-editor run_md_editor.py
```

Copy the generated `dist/md-editor` folder as the portable app folder. The app stores its default database under `data/library.sqlite3` inside that folder.
