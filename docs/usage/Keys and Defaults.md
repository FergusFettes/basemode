# Keys and Defaults

`basemode` supports both environment variables and a persistent local key store.

## Persistent key storage

Keys are saved to:

- `~/.config/basemode/auth.json`

File mode is restricted to user-only (`0600`).

Use:

```bash
basemode keys set openai
basemode keys list
basemode keys get openai
```

## Load order

At startup, settings are loaded in this order (later wins):

1. `~/.config/basemode/auth.json`
2. `.env` (project-local if present)
3. Existing process environment variables

Environment variables are never overwritten by stored keys.

## Default model

Set a default model once:

```bash
basemode default claude-sonnet-4-6
```

Show current default:

```bash
basemode default
```

Clear it:

```bash
basemode default --unset
```

When unset, CLI generation defaults to `gpt-4o-mini`.
