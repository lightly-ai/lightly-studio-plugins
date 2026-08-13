# Zero-Shot Classification Plugin

Zero-shot image classification built on the embeddings Lightly Studio has already computed for your collection. You define the class vocabulary as a table of text prompts and the label each prompt should assign — no training and no fixed label set.

Each image is scored against every prompt, and the label of the best-matching prompt is written as a classification annotation.

The plugin does not run its own vision model. It reads the stored image embeddings and encodes your prompts with the same model that produced them, so a run costs one text encoding per prompt and no image processing at all.

## Setup

### 1. Install the plugin

```bash
uv pip install "git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/clip_zero_shot_classification/"
```

### 2. Embed the collection

The collection must be embedded before the plugin can classify it, which Lightly Studio does when the dataset is ingested. The embedding model is whichever one the collection was embedded with — MobileCLIP (`mobileclip_s0`) by default, configurable through `LIGHTLY_STUDIO_EMBEDDINGS_MODEL_TYPE`. Running on a collection without embeddings reports an error rather than embedding it for you.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompts` | table | dog / cat example rows | Table of `prompt` and `label` pairs — see below |
| `annotation_source` | string | `"clip_zero_shot"` | Target annotation source where predictions are stored. Override this to store results separately from an earlier run |

### The `prompts` table

| Column | Required | Description |
|---|---|---|
| `prompt` | yes | Text prompt, e.g. `"a photo of a dog"`. A row with an empty prompt is an error, not a skipped row |
| `label` | no | Label to assign when this prompt scores highest. Defaults to the prompt itself |

Add and remove rows in the GUI to define your classes. Prompt wording matters — `"a photo of a dog"` typically works better than a bare `"dog"`.

Rows sharing a label merge into a single annotation class, so `a photo of a wolf` and `a photo of a dog` can both map to `canine`.

## Notes

Every image with a stored embedding is labelled with its best-matching prompt. Scores say which prompt fits best, not whether any of them fits at all.

- An image whose true class is not in the table is still labelled with the closest prompt, so the vocabulary has to cover your data. Add a catch-all row such as `a photo of something else` if it does not.
- Scores are a softmax over the prompts you supply, so they are relative to the whole list and adding or removing one prompt shifts all the others.
- Prompts run through the collection's own embedding model, so results match what the same text finds in Lightly Studio's similarity search.

Each image gets at most one `classification` annotation. Images without a stored embedding are skipped.
