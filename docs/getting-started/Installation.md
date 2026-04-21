# Installation

## From PyPI

```bash
pip install basemode
```

Requires Python 3.11+.

## From source

```bash
git clone https://github.com/fergus/basemode
cd basemode
uv sync
```

## Provider keys

`basemode` uses LiteLLM provider credentials. Set keys in your shell or `.env`:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=...
TOGETHER_API_KEY=...
MOONSHOT_API_KEY=...
XAI_API_KEY=...
ZAI_API_KEY=...
```

You can also persist keys with:

```bash
basemode keys set openai
basemode keys list
```

See [[Keys and Defaults]] for details.
