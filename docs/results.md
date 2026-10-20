# Verified results

[Overview](../README.md) · [Study design](study.md) · [Machine-readable values](../results/main_results.json) · [Evidence](evidence.md)

Two studies use different cohorts and metrics. Their scores must not be combined into one accuracy number.

## Original closed-answer VQA

**96 original-image groups · 192 whole-image questions · full training budget.** The metric first averages correctness within each question type by original-image group, then weights the two types equally. B1 and M1 are means across fixed training seeds 17 / 29 / 43. B0 is one deterministic pretrained model.

| Method | Type-balanced accuracy |
|---|---:|
| B0: unadapted BLIP-2 | 37.50% |
| B1: projection adaptation | 76.04% |
| Simple frozen-vision control | 79.69% |
| **M1: joint bridge adaptation** | **83.33%** |

The prespecified **M1 − B1 difference is +7.29 percentage points**, with a paired original-group bootstrap 95% interval of **+0.52 to +13.89**. This supports the stated full-budget closed-answer comparison, subject to the [historical I024 disclosure](study.md#historical-audit-note). The table alone does not establish superiority over every control or every label budget.

[CSV](../results/vqa_summary.csv) · [Comparison CSV](../results/primary_comparisons.csv)

## Lesion-presence transfer

**225 DDR confirmation images · four lesion types · the same existing adapters.** For each lesion, F1 is averaged over its positive and negative classes. Those four scores are averaged, then the three adapter-seed scores are averaged. This endpoint is **mean binary macro-F1**, not accuracy and not positive-class F1 alone.

| Method | Mean binary macro-F1 |
|---|---:|
| B0: unadapted BLIP-2 | 35.19% |
| B1: projection adaptation | 37.79% |
| **M1: joint bridge adaptation** | **55.08%** |

The prespecified **M1 − B1 difference is +17.29 percentage points**, with a paired original-group bootstrap 95% interval of **+13.79 to +20.31**. The conclusion applies to this aggregate reference-presence endpoint; it does not assert that all four lesions, all-positive detection measures, or all-lesion correctness improve uniformly.

[CSV](../results/lesion_summary.csv) · [Class support and exact values](../results/main_results.json)

## How to read the intervals

Both comparisons use **2,000 paired resamples of original-image groups**. Each resample is shared across all fixed model seeds; per-seed metrics are averaged before the method difference is calculated. These intervals quantify sample uncertainty conditional on the trained models. They do not incorporate every source of training randomness. Sample standard deviation across seeds is separately recorded in the CSV/JSON files.

The public tables are a concise selection of verified endpoints. Full representation and evidence-staging experiments belong to separate diagnostic questions and do not establish clinical reasoning here. This public research-code release is not a clinical validation.
