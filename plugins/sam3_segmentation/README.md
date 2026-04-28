# SAM3 Segmentation Plugin

Automatic instance segmentation using [SAM3](https://huggingface.co/facebook/sam3-base) with a text prompt. Runs on image collections in Lightly Studio.

## Setup

### 1. Request access to the model

Visit [facebook/sam3](https://huggingface.co/facebook/sam3) on HuggingFace and request access.

### 2. Authenticate with HuggingFace

```bash
hf auth login
```

Paste your HuggingFace token when prompted. Generate one at https://huggingface.co/settings/tokens (needs read access).

### 3. Install the plugin

```bash
uv pip install -e /path/to/lightly-studio-plugins/plugins/sam3_segmentation
```

### 4. GPU (optional)

By default the plugin runs on CPU. To use a CUDA GPU, reinstall PyTorch with the appropriate CUDA build:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then enable the `use_gpu` parameter in the plugin UI.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_id` | string | `"facebook/sam3"` | HuggingFace model ID — `facebook/sam3` or `facebook/sam3.1` |
| `prompt` | string | `"person"` | Text describing what to segment (e.g. `"car"`, `"dog"`) |
| `confidence_threshold` | float | `0.5` | Minimum score to keep a prediction |
| `use_gpu` | bool | `false` | Run on GPU (CUDA) if available |
