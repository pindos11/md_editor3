from __future__ import annotations

from pathlib import Path

from .db import Repository
from .paths import default_db_path


def main() -> None:
    try:
        import wx
    except ImportError as exc:
        raise SystemExit("wxPython is required to run the GUI. Install with: python -m pip install wxPython") from exc

    from .ui.main_frame import MainFrame

    app = wx.App(False)
    repo = Repository(default_db_path())
    frame = MainFrame(None, repo, default_db_path())
    frame.Show()
    try:
        app.MainLoop()
    finally:
        repo.close()


if __name__ == "__main__":
    main()
