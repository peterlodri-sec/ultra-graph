# ultragraph MCP server

A tiny [Model Context Protocol](https://modelcontextprotocol.io) server that exposes
the `ultragraph` 1-bit (ternary) LLM to any MCP client. It serves the deployed
**Anonymus GPT** — a byte-level ternary GPT trained on the *Gesta Hungarorum* of
Anonymus (c. 1200) — plus the byte tokenizer.

Built with the official [`mcp`](https://pypi.org/project/mcp/) Python SDK
(`FastMCP`) over the **SSE** transport. One file, no extra abstractions:
[`server.py`](./server.py).

## Install

```bash
uv sync --extra mcp
```

## Run

```bash
uv run --extra mcp python mcp_server/server.py
```

The SSE endpoint is served at:

```
http://127.0.0.1:8000/sse
```

> The first call to `anonymus_generate` / `ultragraph_info` loads the deployed
> checkpoint from `examples/data/anonymus.gpt.npz` (cached for the process
> lifetime). If it is missing, train it first with
> `python examples/anonymus_lm.py`.

## MCP client config

```json
{"mcpServers":{"ultragraph":{"url":"http://127.0.0.1:8000/sse"}}}
```

## Tools

| Tool | Description |
| --- | --- |
| `anonymus_generate(prompt, n_new=120, temperature=0.8, top_p=0.9, seed=0)` | Generate text with the deployed 1-bit ternary Anonymus GPT and return the decoded string (prompt + generated). |
| `ultragraph_info()` | Return `{version, deployed_checkpoint_exists, n_params}` (`n_params` is `null` when no checkpoint is present). |
| `tokenize_preview(text)` | Show how the byte tokenizer encodes `text`: `{n_bytes, ids_head}` (first 32 byte ids). |
