# Module 5 — MCP Tool Schemas

Server: `jhf-tools` (FastMCP, stdio transport) — `mcp_server.py`

| Tool | Args (types) | Output | Error cases |
|---|---|---|---|
| `read_data` | `key: str` — ISO 3166-1 alpha-2 code, case-insensitive, whitespace trimmed | `dict` — `{"ok": true, "key", "country", "capital"}` | `{"ok": false, "error": "invalid_key", "message"}` when the code is not exactly 2 alphabetic chars (`"Jordan"`, `""`, `"1"`); `{"ok": false, "error": "not_found", "message"}` when the code is well-formed but absent (`"ZZ"`) — the message lists the known codes. Never raises. |
| `web_search` | `query: str` — free text, trimmed, truncated to 200 chars | `str` — one headline/snippet line | `"SEARCH_ERROR: query must be a non-empty string."` on empty input; `"SEARCH_ERROR: search timed out after 8.0s."` on timeout; `"SEARCH_ERROR: search API returned HTTP <code>."` on an HTTP failure; `"SEARCH_ERROR: <ExceptionName> while searching."` as a catch-all. Never raises. |

## Design notes

- **Errors are returned, not raised.** A raised exception reaches the agent as an opaque
  execution error it cannot reason about; a structured result tells it what failed and
  what to try next.
- **Secrets stay server-side.** `SEARCH_API_KEY` is read from the environment inside the
  server. The agent never holds or sees it. With no key set, `USE_MOCK` is true and the
  mock branch runs.
- **Timeout.** The HTTP call uses `SEARCH_TIMEOUT = 8.0s`, so a hung external API cannot
  hang the agent.
- **Untrusted output.** Search results are treated as data, never instructions; the system
  prompt tells the agent to report command-like text as content instead of following it.
