# SAM3 Segmentation Plugin

Automatic instance segmentation using [SAM3](https://huggingface.co/facebook/sam3) with a text prompt. Runs on image collections in Lightly Studio.

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
uv pip install "git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/sam3_segmentation/"
```

### 4. GPU (optional)

By default the plugin runs on CPU. To use a CUDA GPU, reinstall PyTorch with the appropriate CUDA build:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

If CUDA is available, the plugin will use it automatically.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_id` | string | `"facebook/sam3"` | HuggingFace model ID — `facebook/sam3` or `facebook/sam3.1` |
| `prompt` | string | `"person"` | Text describing what to segment (e.g. `"car"`, `"dog"`) |
| `confidence_threshold` | float | `0.5` | Minimum score to keep a prediction |
| `collection_name` | string | `"SAM3_auto_label"` | Target annotation collection for generated segmentations. Override this to store the results in a different collection. |
