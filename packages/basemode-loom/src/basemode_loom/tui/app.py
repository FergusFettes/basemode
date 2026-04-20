from textual.app import App

from ..session import LoomSession
from .screens.loom import LoomScreen


class BasemodeApp(App[None]):
    """Basemode loom TUI."""

    TITLE = "basemode-loom"
    CSS_PATH = "app.tcss"

    def __init__(self, session: LoomSession) -> None:
        super().__init__()
        self.session = session

    def on_mount(self) -> None:
        self.push_screen(LoomScreen(self.session))

    def on_unmount(self) -> None:
        self.session.save()
