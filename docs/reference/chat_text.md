# chat_text

`basemode.chat.chat_text`

```python
async def chat_text(
    prompt: str | list[dict],
    model: str = "gpt-4o-mini",
    *,
    system: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    transcript: bool | None = None,
    observation: ObservationContext | None = None,
    on_usage: UsageCallback | None = None,
    **extra,
) -> AsyncGenerator[str, None]
```

Stream a model's chat answer token-by-token.

## Notes

- `prompt` is a single user message or a list of `{"role", "content"}` messages.
  `system` is prepended unless the list already opens with a system message.
- The model name is normalized and the request goes through
  `strategies.compat.build_kwargs`, so temperature, thinking-budget and
  `max_completion_tokens` quirks apply exactly as for continuations. No
  continuation prompt or healing is applied.
- Models litellm lists as `mode: completion` have no chat endpoint. They get a
  `User:`/`Assistant:` transcript on the text-completion endpoint, stopped
  (provider-side and client-side) at the next `\nUser:`. `transcript=True`
  forces this, `transcript=False` forces chat.
- Recorded in the observation ledger with strategy `chat` or `chat_transcript`.
  `observation` and `on_usage` behave as in [[continue_text]].
- `extra` is forwarded to the provider request.
- Raises `EmptyCompletionError` (`strategy="chat"` or `"completion"`) when the
  model returns no content.
