# OpenRouter Image Captioning Plugin

Generates image captions with any vision-capable model available through
[OpenRouter](https://openrouter.ai/) and stores them as native Lightly Studio captions.

## Setup

### 1. Create an OpenRouter API key

Sign up at <https://openrouter.ai/>, add credits, then create a key at
<https://openrouter.ai/keys>.

### 2. Export the key

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Lightly Studio also loads a `.env` file from its working directory, so
`OPENROUTER_API_KEY=sk-or-v1-...` in `.env` works too. The key is read on every run, but
the server process must already have it, so restart Lightly Studio after setting it.

### 3. Install the plugin

```bash
uv pip install "git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/openrouter_image_captioning/"
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | `"qwen/qwen3-vl-8b-instruct"` | OpenRouter model slug of a vision-capable model. Browse and compare at [openrouter.ai/models](https://openrouter.ai/models) |
| `prompt` | string | see below | Instruction sent to the model alongside the image |
| `base_url` | string | `"https://openrouter.ai/api/v1"` | OpenAI-compatible base URL. Change only to use a proxy or a different gateway |
| `skip_captioned` | bool | `true` | Skip images that already have at least one caption |
| `max_samples` | int | `200` | Maximum images per run. `0` means no limit |
| `max_image_edge` | int | `256` | Longest edge in pixels after downscaling. Lower is cheaper and faster. Minimum 64, or `0` to disable resizing |
| `max_tokens` | int | `200` | Caption length cap in tokens |
| `temperature` | float | `0.2` | Sampling temperature. `0.0` for the most reproducible captions |
| `max_concurrency` | int | `16` | Images captioned in parallel. Max 64 |
| `provider_sort` | string | `"throughput"` | How OpenRouter picks between providers serving the model: `throughput`, `latency`, `price`, or empty for its default load balancing |
| `request_timeout` | float | `60.0` | Per-request timeout in seconds |
| `max_retries` | int | `3` | Retries per image on rate limits (429), server errors (5xx) and network errors |

The default prompt is:

> Describe this image in one or two concise sentences. Name the main objects, their
> notable attributes and the overall scene. Do not begin with 'The image shows'.

## Choosing a model

Browse the full catalogue at **<https://openrouter.ai/models>**. Filter by image input to
see only vision-capable models, and compare price and throughput there before switching:

- Models page (filter to image input): <https://openrouter.ai/models?modality=text+image-%3Etext>
- Machine-readable list, useful for checking a slug exists: <https://openrouter.ai/api/v1/models>

The default `qwen/qwen3-vl-8b-instruct` is a small, cheap, fast vision model that is a good
fit for short captions over large datasets. A text-only model either returns an error or
ignores the image and invents a caption, so make sure whatever you pick accepts images.

## Making a large run faster

Requests are issued concurrently, so wall-clock time is roughly
`image_count / max_concurrency x per-request latency`. In order of impact:

1. **Raise `max_concurrency`** (up to 64). This is the main lever. If the result message
   starts reporting failures you have hit OpenRouter's per-key rate limit, so come back
   down. Retries honour `Retry-After`, so a burst degrades into backoff rather than errors,
   which costs the time you were trying to save.
2. **Set `provider_sort`**. A model is often served by several providers at very different
   speeds. `throughput` (the default) favours bulk speed; `latency` favours the fastest
   first token. Note that sorting disables OpenRouter's load balancing and tries providers
   in order.
3. **Lower `max_image_edge`**. Images dominate the prompt, so this cuts both latency and
   cost. The default of `256` is already aggressive; raise it toward `512`-`768` if captions
   start missing small objects or text.

`max_tokens` is only a cap, not a target, so lowering it does not speed up a run unless
the model would otherwise have kept going past it.

OpenRouter's [Batch API](https://openrouter.ai/docs/batch-quickstart) cannot be used here:
it does not accept multimodal input, and its completion window is 24 hours.

## Finding a run in OpenRouter

Every run sends a `session_id` that groups all of its requests, so one captioning run
shows up as a single logical unit in OpenRouter's activity view:

```text
lightly-studio-captioning-20260729T150312Z-a1b2c3
```

The prefix separates these requests from other tools sharing the API key, the UTC
timestamp is the run's start time, and the random suffix keeps two runs started in the
same second apart. The id and the run timings are written to the server log:

```text
INFO Captioning 200 image(s) with qwen/qwen3-vl-8b-instruct at concurrency 16 as
     OpenRouter session 'lightly-studio-captioning-20260729T150312Z-a1b2c3'.
INFO Captioned 198/200 image(s) in 41.3s (4.8 image/s). Request time median 0.9s,
     slowest 3.4s.
```

Read those two numbers together when tuning. Throughput should land near
`max_concurrency / median request time` — 16 / 0.9 s ≈ 17 image/s above. Getting far less
than that means something other than the model is the bottleneck, usually retries after
rate limiting. If the median request time itself is high, the model or provider is slow, so
try a different `model` or `provider_sort`. Per-image timings are available at `DEBUG`.

OpenRouter also treats `session_id` as a sticky routing key, preferring to send a
session's requests to one provider. That costs nothing here, because separate images
share no cacheable prompt prefix and `provider_sort` already disables load balancing.

## Notes

- Captions are written to Lightly Studio's caption collection, so they appear in the
  caption view and in the "has captions" filter immediately.
- Operators run synchronously on the API request, so a run blocks until it finishes, and
  closing the browser does not cancel it. Keep `max_samples` modest and raise
  `max_concurrency` rather than launching one huge batch.
- **Every run costs money**, per image and per token. Check pricing at
  <https://openrouter.ai/models> and start with a small `max_samples`.
- Images are downscaled and JPEG re-encoded before upload. Originals are never modified.
- `:free` model variants are heavily rate limited and will spend most of a run in backoff.
- The `model` must accept image input. A text-only model either returns an error or
  ignores the image and makes something up.
- With `skip_captioned = false` a second run **adds** a caption, it does not replace the
  existing one.
- Failures are isolated per image: unreadable files and rejected requests are counted and
  reported, and the rest of the run continues.
