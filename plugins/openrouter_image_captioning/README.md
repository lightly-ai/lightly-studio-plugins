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

A `.env` file in the Lightly Studio working directory works too. Restart Lightly Studio
after setting the key.

### 3. Install the plugin

```bash
uv pip install "git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/openrouter_image_captioning/"
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | `"qwen/qwen3-vl-8b-instruct"` | Slug of a vision-capable model. Browse and compare at [openrouter.ai/models](https://openrouter.ai/models?modality=text+image-%3Etext) |
| `prompt` | string | see below | Instruction sent to the model alongside the image |
| `max_samples` | int | `200` | Maximum images per run. `0` means no limit |
| `max_image_edge` | int | `256` | Longest edge in pixels after downscaling. Lower is cheaper and faster. Minimum 64, or `0` to disable resizing |
| `max_tokens` | int | `200` | Caption length cap in tokens |
| `temperature` | float | `0.2` | Sampling temperature. `0.0` for the most reproducible captions |
| `max_concurrency` | int | `16` | Images captioned in parallel. Max 64 |

The default prompt is:

> Describe this image in one or two concise sentences. Name the main objects, their
> notable attributes and the overall scene. Do not begin with 'The image shows'.

The request timeout (60s), retry count (3) and OpenRouter provider sort (`throughput`)
are fixed in [`settings.py`](src/lightly_plugins_openrouter_image_captioning/settings.py).

The operator captions whatever the current view holds. Use the Lightly Studio filters to
choose which images that is — filtering to images without captions is how you avoid
captioning an image twice.

## Tuning a large run

Wall-clock time is roughly `image_count / max_concurrency x per-request latency`, so
`max_concurrency` is the main lever — raise it until the result message starts reporting
failures, which means you have hit the per-key rate limit. Lowering `max_image_edge` cuts
both latency and cost.

Each run logs its OpenRouter `session_id` and timings to the server log:

```text
INFO Captioned 198/200 image(s) in 41.3s (4.8 image/s). Request time median 0.9s,
     slowest 3.4s.
```

Throughput should land near `max_concurrency / median request time`. Far below that means
retries after rate limiting; a high median means the model or provider is slow. Per-image
timings are logged at `DEBUG`.

## Notes

- **Every run costs money**, per image and per token. Check pricing at
  <https://openrouter.ai/models> and start with a small `max_samples`.
- The `model` must accept image input. A text-only model either errors or ignores the
  image and invents a caption.
- Runs block until finished and cannot be cancelled by closing the browser. Keep
  `max_samples` modest and raise `max_concurrency` instead of launching one huge batch.
- Images are downscaled and JPEG re-encoded before upload. Originals are never modified.
- `:free` model variants are heavily rate limited and spend most of a run in backoff.
- Captioning an image that already has a caption **adds** a second one rather than
  replacing it. Filter the view to images without captions before re-running.
- Failures are isolated per image: unreadable files and rejected requests are counted and
  reported, and the rest of the run continues.
