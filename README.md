<p align="center">
  <img src="assets/banner.png" alt="Retinal BLIP-2 Visual Question Answering — adapt the bridge, evaluate the evidence" width="1120" />
</p>

<p align="center">
  <strong>Targeted vision–language adaptation for retinal question answering</strong><br />
  Frozen visual and language backbones · Matched model comparisons · Independent lesion evaluation
</p>

<p align="center">
  English · <a href="README.zh-CN.md">简体中文</a> ·
  <a href="docs/study.md">Study</a> ·
  <a href="docs/results.md">Results</a> ·
  <a href="docs/reproducibility.md">Reproduce</a> ·
  <a href="docs/data.md">Data &amp; access</a>
</p>

---

## The idea

Can adapting the image–language connection improve retinal question answering and recognition of independently annotated lesions?

This project adapts **BLIP-2** while freezing its visual encoder and language model. Separate VQA and lesion studies examine **task answers** and **recognition of visual evidence**.

- **Adapt the visual bridge:** compare the original model, projection-only adaptation, and joint Q-Former/projection adaptation.
- **Test on actual annotations:** evaluate lesion presence against four types of reference segmentation masks, with missing masks kept distinct from negative labels.
- **Keep the comparison traceable:** fixed seeds, original-image grouping, frozen evaluation rules, independent metric recalculation, and fresh-checkpoint inference checks.

## Method

<p align="center">
  <picture>
    <source media="(max-width: 640px)" srcset="assets/method-mobile.png" />
    <img src="assets/method.png" alt="BLIP-2 adaptation: frozen ViT image encoder, trainable Q-Former and projection, frozen OPT language model. Answer-loss gradients flow through frozen OPT to the adapted visual bridge." width="1120" />
  </picture>
</p>

| Model | What is trained | Role |
| :--- | :--- | :--- |
| **B0** | Nothing | Pretrained BLIP-2 baseline |
| **B1** | Image-to-language projection | Projection-only control |
| **M1** | Q-Former, query tokens, and projection | Joint visual-bridge adaptation |

The language-model parameters stay frozen, but its forward pass retains the gradient path to the trainable visual prefix. The image encoder stays frozen throughout. [Implementation and evaluation design →](docs/study.md)

## Selected findings

<p align="center">
  <picture>
    <source media="(max-width: 640px)" srcset="assets/results-mobile.png" />
    <img src="assets/results.png" alt="Original VQA: B1 76.04%, M1 83.33% type-balanced accuracy on 96 image groups and 192 questions. Independent lesion study: B1 37.79%, M1 55.08% binary macro-F1 on 225 images. Three-seed means; paired group-bootstrap intervals shown in the chart." width="1120" />
  </picture>
</p>

| Evaluation | B1 | M1 |
| :--- | ---: | ---: |
| Retinal VQA · accuracy | 76.04% | **83.33%** |
| Lesion recognition · macro-F1 | 37.79% | **55.08%** |

**Separate cohorts and metrics:** 96 original-image groups / 192 VQA questions; 225 images / four lesion types. Both report means over seeds 17, 29, 43. Lesion macro-F1 averages positive/negative F1 within each lesion, then across lesions. Paired 95% intervals use 2,000 original-image-group bootstrap resamples.

[Metric definitions, comparison scope, and evidence →](docs/results.md)

Feature-probe analyses and staged evidence diagnostics are documented separately in the [study notes](docs/study.md). The staged tasks use declared count/Boolean rules; clinical conditional reasoning requires complete, independently verified medical evidence and answers.

## Get started

```bash
python -m pip install \
  -e '.[dev,figures]'
python -m pytest -q
```

The [reproduction guide](docs/reproducibility.md) covers legal data/model acquisition, local paths, and training/evaluation commands. Code, configurations, aggregate results, and figure sources are included; medical images, restricted annotations, and weights are acquired separately.

```bash
# Rebuild the README graphics.
python tools/build_figures.py
```

## Explore the repository

| Resource | What it contains |
| :--- | :--- |
| [Study](docs/study.md) | Adaptation, data roles, probes, and staged evidence evaluation |
| [Results](docs/results.md) | Metric definitions, sample counts, and supported comparisons |
| [Data & access](docs/data.md) | Official sources, annotation semantics, and legal acquisition |
| [Reproducibility](docs/reproducibility.md) | Environment, configurations, and executable workflows |
| [Evidence](docs/evidence.md) | Source hashes, independent checks, and release provenance |
| [Machine-readable results](results/main_results.json) | The aggregate values used in the README and charts |

## Credits & terms

Built on [BLIP-2](https://proceedings.mlr.press/v202/li23q.html) and the [Salesforce BLIP-2 model release](https://huggingface.co/Salesforce/blip2-opt-2.7b), with retinal tasks and annotations from the sources documented in [Data & access](docs/data.md).

See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY.md) for the applicable code, model, and data terms. This repository presents an experimental adaptation and evaluation study; clinical validation remains outside the established evidence.
