# Deployment

## Local Gradio application

```bash
uv sync --locked
mkdir -p models
# Place the verified production checkpoint at models/roadguard-production.pt
uv run python app.py
```

Set `ROADGUARD_WEIGHTS=/absolute/path/to/checkpoint.pt` to use another compatible four-class checkpoint. Startup is lazy: the dashboard can render without weights, but inference fails explicitly until genuine compatible weights are present.

## Hugging Face Space

The planned free deployment is a Gradio Space containing this repository plus the inference-only production checkpoint. `requirements.txt` installs the package in editable mode. The Space must expose the full AGPL-3.0 source, attribute RDD2022, and avoid bundling raw dataset files.

CPU hosting may make video processing impractical. The public feature matrix must be verified after deployment; local-only features will be labeled rather than advertised online. A Space is not considered live until an unauthenticated fresh session successfully loads the model, runs a real image, aligns boxes to the original image, handles empty output, and downloads an advertised export.

## Model repository

The separate model repository should contain only:

- stripped inference checkpoint;
- optional verified ONNX export;
- class mapping and input size;
- production-selection record and genuine test metrics;
- checkpoint SHA-256;
- model card, limitations, dataset citation, and AGPL-3.0 license notice.

Optimizer states, raw RDD2022 images, credentials, and local paths must not be uploaded.

## ONNX parity gate

ONNX is optional. It is released only after representative-image comparison against PyTorch records class agreement, matched-box IoU, confidence differences, and end-to-end latency. An export file existing is not evidence of browser/runtime compatibility.

