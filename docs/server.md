# OpenAI-compatible HTTP server

`valkyr --serve <model>` runs an HTTP server that speaks a strict
subset of OpenAI's `/v1/chat/completions` and `/v1/models` API.
LangChain, Cline, Aider, the official `openai` Python client, and
anything else that posts to chat-completions endpoints work out of
the box — no shim layer.

The server is a thin adapter over the same [`InferenceRunner`](
embedding.md#three-api-tiers) the embed path uses. JSON in, JSON
or SSE out, no business logic between the wire and the inference
core.

## Quick start

```bash
# Boot a server.
valkyr --serve Qwen/Qwen3-4B-Instruct-2507 --q4k --port 8080 --id qwen3-4b

# Non-streaming
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-4b",
    "messages": [
      {"role": "user", "content": "Tell me a short joke."}
    ],
    "max_tokens": 64
  }'

# Streaming (SSE)
curl -N -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "Write me a haiku."}],
    "stream": true,
    "max_tokens": 48
  }'

# Models list
curl -s http://127.0.0.1:8080/v1/models
```

## Use it from the official `openai` Python client

```python
from openai import OpenAI

c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

# Multi-turn, non-streaming.
r = c.chat.completions.create(
    model="qwen3-4b",
    messages=[
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me a short joke."},
    ],
    max_tokens=64,
)
print(r.choices[0].message.content)
print(f"tokens={r.usage.total_tokens}  finish={r.choices[0].finish_reason}")

# Streaming.
stream = c.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "Tell me a haiku."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Validated against `openai==1.98.0` end-to-end (multi-turn,
streaming, error-shape compliance, `/v1/models`).

## CLI flags

```
valkyr --serve <model> [--port N] [--bind ADDR] [--id PUBLIC_ID]
                      [--max-new N] [--q4|--q4k]
```

| Flag | Default | Notes |
|---|---|---|
| `<model>` | — | HF id or directory used to load the weights. |
| `--id` | `<model>` | Public model id surfaced via `/v1/models` and validated against the request's `model` field. Letting clients post the HF id verbatim works; setting a short id (e.g. `qwen3-4b`) is friendlier. |
| `--port` | `8080` | TCP port to bind. |
| `--bind` | `127.0.0.1` | Bind address. v0 has **no auth** — pass `0.0.0.0` only on a trusted network. |
| `--max-new` | `256` | Default `max_tokens` cap when a request omits it. |
| `--q4` / `--q4k` | bf16 | Per-layer projection weight quantization. `--q4k` is the production default for the matmul hot path; `--q4` keeps llama.cpp Q4_0 compatibility. lm_head and embeddings stay bf16 either way. |

## Endpoints

### `POST /v1/chat/completions`

The headline endpoint. Body is OpenAI's `chat.completion` request
schema (subset — see "Field support" below).

**Non-streaming response** (`stream: false` or absent):

```json
{
  "id": "chatcmpl-1",
  "object": "chat.completion",
  "created": 1714780800,
  "model": "qwen3-4b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hi!"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 17,
    "completion_tokens": 3,
    "total_tokens": 20
  }
}
```

**Streaming response** (`stream: true`): Server-Sent Events. First
frame carries `delta.role: "assistant"`, subsequent frames carry
`delta.content`, the last frame carries `finish_reason`, then the
literal `data: [DONE]\n\n` terminator closes the stream.

```
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1714780800,"model":"qwen3-4b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1714780800,"model":"qwen3-4b","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1714780800,"model":"qwen3-4b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### `GET /v1/models`

Returns the single loaded model:

```json
{
  "object": "list",
  "data": [{
    "id": "qwen3-4b",
    "object": "model",
    "created": 1714780800,
    "owned_by": "valkyr"
  }]
}
```

Future enhancement: enumerate all cached HF models (the cache
discovery is already wired via `hf_cache.listModels()`); v0 just
shows the loaded one.

### `OPTIONS *`

CORS preflight. Returns 204 with permissive
`Access-Control-Allow-*` headers so browser clients can hit the
endpoint cross-origin.

## Field support

### Honored

| Field | Behavior |
|---|---|
| `model` | Must equal the loaded `--id`. Mismatch → 404 `model_not_found`. |
| `messages` | Array of `{role, content}` (or content-parts). Roles: `system`, `user`, `assistant`. Composed through the family's chat template. |
| `messages[].content` | String OR array of `{type: "text", text: "..."}` parts (concatenated). |
| `max_tokens` | Cap on completion length. |
| `stream` | `true` → SSE; `false` or absent → JSON. |
| `stop` | String or array of ≤4 strings. Trailing-text suffix matched after each emitted token. |
| `seed` | Accepted; ignored (greedy sampler is deterministic). |
| `n` | Must be `1`. Anything else → 400. |
| `user` | Accepted, opaque, logged in `corr` mapping but otherwise ignored. |

### Silently ignored (no error)

`temperature`, `top_p`, `presence_penalty`, `frequency_penalty`,
`logit_bias`, `response_format`, `logprobs`, `tools`,
`tool_choice`. Sampler knobs land alongside the existing samplers
TODO (`SamplerKind` union in `session.zig`). Tools are deferred.

### Rejected with 400

- `n != 1` — only one completion per request supported.
- `stop` array > 4 entries — matches OpenAI's spec limit.
- `messages[].content` containing `image_url` / `image_bytes` /
  audio parts — the protocol carries an `Attachment` placeholder
  that's reserved for vision in v2; v0 only routes `.text`.
- `role: "tool"` — function calling deferred.
- Malformed JSON, missing `model` or `messages`.

### Errors

OpenAI-format error envelope:

```json
{
  "error": {
    "message": "model not found; only the loaded model is served",
    "type": "model_not_found"
  }
}
```

| HTTP status | `type` | When |
|---|---|---|
| 400 | `invalid_request_error` | Malformed JSON, `n != 1`, image content parts, missing required fields. |
| 404 | `model_not_found` | `model` doesn't match `--id`, or unknown route. |
| 503 | `internal_error` | Runner queue full (`error.QueueFull`). |
| 500 | `internal_error` | Runner emitted a terminal err event mid-decode. |

## How `finish_reason` maps

| Runner reason | `finish_reason` | When |
|---|---|---|
| `stop` | `"stop"` | Hit the family's EOT token (e.g. `<\|im_end\|>`, `<\|eot_id\|>`, `<end_of_turn>`) or matched a `stop` string. |
| `length` | `"length"` | Hit `max_tokens`. |
| `cancelled` | `"stop"` | Reserved. (Cancel commands aren't surfaced through the HTTP API today; HTTP clients close the connection instead.) |
| `timeout` | `"stop"` | Reserved. |
| `server_shutdown` | `"stop"` | Reserved. |

The EOT token's decoded form (`<|im_end|>`, `<|eot_id|>`,
`<end_of_turn>`) is **suppressed by the Runner** — it never
appears in `content`. Earlier versions leaked it; we filter at
the on_token bridge so completion text is clean.

## Architecture

```
                main thread
                ─ parses args, loads model, builds Session/Recorder/Runner
                ─ Server.start                   accept loop ──┐
                                                                │
                                                                ▼
                                                       connection thread
                                                          (one per client)
                                                          ─ parse JSON
                                                          ─ runner.submit ────┐
                                                          ─ wait on mailbox   │
                                                          ─ stream/JSON       │
                                                                              ▼
                                              fan-out thread
                                              ─ runner.pollEvent (sole consumer)
                                              ─ dispatch by event.corr to mailbox

                                                                              ┌─ runner thread
                                                                              │   (Sketch #3)
                                                                              ─ tickInline loop
                                                                              ─ owns Session + GPU
```

**Concurrency:** thread-per-connection on accept. Connection
threads serialize behind a `submit_mu` mutex when calling
`runner.submit` (preserves the SPSC contract on the runner's
command queue). The fan-out thread is the sole consumer of
`runner.pollEvent`, demuxing events to per-`corr` mailboxes via
condition variables. Connection threads block on their own
mailbox CV.

**Serialization at the runner level:** v0 has one active inference
request at a time. Concurrent HTTP clients queue cleanly through
the runner's backlog — they get accepted in order; the second
request waits on its mailbox until the first finishes. Inference
is the bottleneck, so the queue model matches the hardware reality.

## Why bare-metal HTTP/1.1

The server uses `std.net` directly rather than `std.http.Server`.
Stdlib HTTP has churned across Zig 0.11 → 0.12 → 0.13 → master
(deprecated, rebuilt, SSE awkward to thread through), and the
HTTP/1.1 grammar we need is a strict subset (request line +
headers up to `\r\n\r\n` + Content-Length-bounded body, no
chunked-encoding on the request side, no keep-alive). ~400 lines
of `src/server/http.zig` covers it; we control the SSE flush
cadence directly.

## What we don't do (yet)

| | v1 (next) | v2 (later) |
|---|---|---|
| **Auth** | `--require-token X` bearer-token gate | API-key registry + per-key rate limits |
| **Multi-session** | Multiple concurrent KV caches keyed by `corr` | Session pool + LRU eviction |
| **Samplers** | `temperature`, `top_p`, `top_k` — already wired in `SamplerKind`, just need implementation | Speculative decoding (Qwen3.6 MTP head) |
| **Endpoints** | — | `/v1/embeddings`, `/v1/completions` (legacy) |
| **Vision** | — | `image_url` / `image_bytes` content parts route to a vision encoder before `appendMessages` |
| **Tool calls** | — | `tools`, `tool_choice`, `tool_calls` finish reason |
| **Concurrency** | Server-level mutex sufficient for typical loads | True multi-stream batching at the GPU layer |

The protocol shape is already future-proofed: `Attachment` carries
`.image_url` / `.image_bytes` variants that wire-time clients can
already populate (they just bounce with a clean error today). When
we add vision the wire format doesn't change.

## Pointers

- `src/server/http.zig` — bare-metal HTTP/1.1 reader + response /
  SSE writer.
- `src/server/json.zig` — `parseChatRequest`, `writeChatResponse`,
  `buildStreamChunk`, `writeError`, `writeModelsResponse`.
- `src/server/server.zig` — `Server` struct, accept loop, fan-out
  thread, connection handler.
- `src/main.zig` — `--serve` flag handler (`runServe`).

## Validation

Manually validated against:
- curl POST + `--no-buffer` SSE
- official `openai` Python client v1.98 (chat.completions.create
  streaming + non-streaming, models.list)
- Cross-family: Gemma 2B IT, Llama 3.2 1B-Instruct, Qwen3 4B
  dense, Qwen3.5 0.8B hybrid
- Edge cases: model_not_found, n!=1 reject, image_url reject,
  content-parts text concat, `stop:["..."]` truncation
