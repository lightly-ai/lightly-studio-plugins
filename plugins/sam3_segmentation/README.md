# SAM3 Plugin

Automatic instance segmentation using [SAM3](https://huggingface.co/facebook/sam3) with a table of text prompts. Runs on image collections in Lightly Studio.

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

The plugin picks a device automatically: CUDA, else MPS, else CPU. To use a CUDA GPU, reinstall PyTorch with the appropriate CUDA build:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_id` | string | `"facebook/sam3"` | HuggingFace model ID — `facebook/sam3` or `facebook/sam3.1` |
| `prompts` | table | `person` | Prompts to segment with and the labels to assign. See below |
| `confidence_threshold` | float | `0.5` | Minimum score to keep a prediction. Applies to every prompt |
| `bounding_boxes_only` | bool | `false` | Store bounding boxes instead of segmentation masks |
| `collection_name` | string | `"SAM3_auto_label"` | Target annotation collection for generated segmentations. Override this to store the results in a different collection. |

### The `prompts` table

| Column | Required | Description |
|---|---|---|
| `prompt` | yes | What to segment, e.g. `"person"`. Must be at most 32 tokens — SAM3's text encoder rejects anything longer |
| `label` | no | Annotation label for this prompt's detections. Leave empty to use the prompt |

Add rows to segment several concepts in one run. Rows sharing a label merge into a single
annotation class, so `car` and `truck` can both map to `vehicle`.

## Notes

- SAM3 computes masks and boxes together. `bounding_boxes_only` selects which of the two is stored, not what the model computes. One annotation is created per detected object either way:
  - unticked (default): `segmentation_mask` annotations carrying the mask *and* its bounding box.
  - ticked: `object_detection` annotations with a bounding box.
- A `segmentation_mask` annotation always stores its bounding box. So the default already gives you mask *and* box on a single annotation. Tick `bounding_boxes_only` only when you want plain detections without the mask.
- Annotations are written to the collection given by `collection_name`.
- Prompts are evaluated independently and overlapping detections are all kept, so an object matched by two prompts yields two annotations.
