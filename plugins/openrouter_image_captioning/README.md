# OpenRouter Image Captioning Plugin

Captions images with any vision-capable model on
[OpenRouter](https://openrouter.ai/) and stores the result as a Lightly Studio caption.

## Setup

Create a key at <https://openrouter.ai/keys> (the account needs credits), export it, then
restart Lightly Studio. A `.env` file in the working directory works too.

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
uv pip install "git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/openrouter_image_captioning/"
```

## Parameters

Out-of-range values are clamped to the nearest usable one.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | `"qwen/qwen3-vl-8b-instruct"` | Slug of a vision-capable model. Browse at [openrouter.ai/models](https://openrouter.ai/models?modality=text+image-%3Etext) |
| `prompt` | string | Describe this image in one or two concise sentences… | Instruction sent alongside the image |
| `max_samples` | int | `200` | Maximum images per run. `0` means no limit |
| `max_image_edge` | int | `256` | Longest edge in pixels after downscaling; images are JPEG re-encoded before upload, never on disk. Lower is cheaper and faster. Minimum 64, or `0` to disable resizing |
| `max_concurrency` | int | `16` | Images captioned in parallel. Max 64 |

Caption length cap (200 tokens), temperature (0.2), request timeout (60s), retries (3) and
provider sort (`throughput`) are fixed in
[`openrouter_client.py`](src/lightly_plugins_openrouter_image_captioning/openrouter_client.py).

## Notes

- **Every run costs money**, per image and per token. Start with a small `max_samples`.
- The operator captions whatever the current view holds. Filter in the GUI to choose which
  images — a caption is **added** to images that already have one, not replaced.
- `max_concurrency` is the main speed lever. Raise it until the result reports failures,
  which means you hit the per-key rate limit. `:free` models are throttled hardest.
- Runs block until finished and cannot be cancelled by closing the browser.
