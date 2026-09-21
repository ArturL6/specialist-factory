# Specialist Factory

A compact Python POC that turns partially labeled text, image, and image-text samples into specialist classifiers. Teachers, aggregation, human labels, and student inference are intentionally separate.

## Architecture

- **Samples:** modality-neutral JSONL (`id`, `text`, `image`, optional `human_label`, `metadata`).
- **Teachers:** protocol-based adapters for BART MNLI, CLIP, a documented OpenRouter VLM adapter, and deterministic offline mocks.
- **Provenance:** each cached teacher signal stores model, source/version, latency, cost, signal type, and request-derived cache key.
- **Aggregation:** weighted mean, log-probability mean, or majority voting, with agreement/confidence/entropy thresholds and abstention.
- **Students:** small local hashed-text and image-stat encoders, explicit late fusion for multimodal data, MLP head, and Lightning soft-label distillation plus human cross entropy.
- **Evaluation:** separate human-label accuracy, student/teacher agreement, and teacher abstention rate.

`self_reported_probability` is preserved as a separate signal type and is not treated as calibrated likelihood.

## Install and run

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e '.[dev]'

python -m specialist_factory generate --config config/text_demo.yaml
python -m specialist_factory annotate --config config/text_demo.yaml
python -m specialist_factory train --config config/text_demo.yaml
python -m specialist_factory evaluate --config config/text_demo.yaml --checkpoint runs/text_demo/best.ckpt
python -m specialist_factory predict --checkpoint runs/text_demo/best.ckpt --text 'Where is my package?'
```

Run the same sequence for `config/vision_demo.yaml` and `config/multimodal_demo.yaml`; `predict` accepts `--image path.png` and optionally `--text`.

The demos use deterministic `mock` teachers so they are CPU-only and reproducible. Swap in the commented HF teacher configurations after installing `.[hf]`; model downloads are runtime dependencies. The external VLM adapter uses OpenRouter's documented [`/api/v1/chat/completions`](https://openrouter.ai/docs/api/reference/overview) request schema and JSON-object response format; it requires an environment key and deliberately stores its confidence as self-reported.

## Limits

This is a vertical POC, not production MLOps: image student features are intentionally tiny CPU features, API response schemas are provider-specific, and calibration/held-out human evaluation are required before any real deployment.
