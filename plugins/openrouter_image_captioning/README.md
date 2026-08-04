# OpenRouter Image Captioning Plugin

Captions images with any vision-capable model on
[OpenRouter](https://openrouter.ai/) and stores the result as a Lightly Studio caption.

## Setup

Install the plugin, then create a key at <https://openrouter.ai/keys> (the account needs
credits).

```bash
uv pip install "git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/openrouter_image_captioning/"
```

There are two ways to give the plugin that key:

- **`api_key` parameter** — paste the key into the operator form. Takes effect immediately,
  with no export and no restart.
- **`OPENROUTER_API_KEY` environment variable** — export it (or put it in a `.env` file in
  the directory where you start Lightly Studio) and restart Lightly Studio. Nothing to type
  per run.

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

When both are set the `api_key` parameter wins, so a key pasted for one run overrides an
exported one.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | `"qwen/qwen3-vl-8b-instruct"` | Slug of a vision-capable model. Browse at [openrouter.ai/models](https://openrouter.ai/models?modality=text+image-%3Etext) |
| `prompt` | string | Describe this image in one or two concise sentences… | Instruction sent alongside the image |
| `api_key` | string | `""` | OpenRouter API key. Blank falls back to `OPENROUTER_API_KEY` |
| `max_image_edge` | int | `256` | Longest edge in pixels after downscaling; images are JPEG re-encoded before upload, never on disk. Lower is cheaper and faster. `0` disables resizing |
| `max_concurrency` | int | `4` | Images captioned in parallel |

Caption length cap (200 tokens), temperature (0.2), request timeout (60s), retries (3) and
provider sort (`throughput`) are fixed in
[`openrouter_client.py`](src/lightly_plugins_openrouter_image_captioning/openrouter_client.py).

## Notes

- The `api_key` field is a **plain, unmasked text input**: the key stays visible on screen
  and is sent in the operator's execute request. Prefer `OPENROUTER_API_KEY` when sharing a
  screen or working on a machine you do not control.
- **Every run costs money**, per image and per token. There is no cap: the operator
  captions **every** image the current view holds, so filter it down first and check the
  image count before starting a run.
- Filter in the GUI to choose which images — a caption is **added** to images that already
  have one, not replaced.
- `max_concurrency` is the main speed lever. Raise it until the result reports failures,
  which means you hit the per-key rate limit. `:free` models are throttled hardest.
- Runs block until finished and cannot be cancelled by closing the browser.
