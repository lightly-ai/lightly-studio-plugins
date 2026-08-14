# Zero-Shot Classification Plugin

Zero-shot classification that uses the embeddings Lightly Studio already computed for your collection. You define the classes as a table of text prompts and the label that each prompt assigns. The plugin scores each sample against every prompt. The label of the best match becomes a classification annotation.

The plugin runs no vision model of its own. It reads the stored embeddings and encodes your prompts with the same model that produced them. One run costs one text encoding per prompt.

The plugin runs on image, video, and video frame collections.

## Setup

### 1. Install the plugin

The plugin needs Lightly Studio 1.0.5 or later.

```bash
uv pip install "git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/zero_shot_classification/"
```

### 2. Embed the collection

The collection must have embeddings before the plugin can classify it. Lightly Studio computes them when it ingests the dataset. The plugin uses the model that the collection was embedded with, set by `LIGHTLY_STUDIO_EMBEDDINGS_MODEL_TYPE`. On a collection without embeddings, the plugin reports an error. It does not embed the collection for you.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompts` | table | dog / cat example rows | Table of `prompt` and `label` pairs. |
| `annotation_source` | string | `"zero_shot"` | Target annotation source where predictions are stored. Override this to store results separately from an earlier run. |

### The `prompts` table

| Column | Required | Description |
|---|---|---|
| `prompt` | yes | Text prompt, for example `"a photo of a dog"`. An empty prompt is an error. The plugin does not skip the row. |
| `label` | no | Label to assign when this prompt scores highest. Defaults to the prompt itself. |

Add and remove rows in the GUI to define your classes. You must give at least two prompts. Wording changes the result: `"a photo of a dog"` usually scores better than `"dog"`.

Rows that share a label merge into one annotation class. Both `a photo of a wolf` and `a photo of a dog` can map to `canine`.

## Notes

Every sample with a stored embedding gets the label of its best-matching prompt. The scores show which prompt fits best. They do not show whether any prompt fits at all.

- A sample whose true class is not in the table still gets the closest prompt. Your prompts must cover your data. If they do not, add a catch-all row such as `a photo of something else`.
- The scores are a softmax over the prompts that you supply. They are relative to the whole list, so each prompt that you add or remove shifts all the others.

Each sample gets one `classification` annotation at most. The plugin skips samples that have no stored embedding.
