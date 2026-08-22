"""Public exceptions raised by continuation strategies."""


class EmptyCompletionError(RuntimeError):
    """A provider stream ended without yielding any content tokens.

    Callers otherwise only see a silently empty continuation, with no way to
    tell a content filter from a truncated response or a provider glitch.
    """

    def __init__(
        self, *, model: str, strategy: str, finish_reason: str | None = None
    ) -> None:
        self.model = model
        self.strategy = strategy
        self.finish_reason = finish_reason
        detail = f", finish_reason={finish_reason!r}" if finish_reason else ""
        super().__init__(f"{strategy!r} strategy got no tokens from {model!r}{detail}")
