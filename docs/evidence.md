# Evidence and release provenance

[Overview](../README.md) · [Results](results.md) · [Study](study.md)

## What is traceable

Every published result has a source identifier, relative archived-artifact name and SHA-256 in [main_results.json](../results/main_results.json). CSV rows repeat their numeric source fingerprint. No private server paths are needed to read the public results.

| Public file | Contents |
|---|---|
| [main_results.json](../results/main_results.json) | Exact means, seed values, metric definitions, denominators, positive primary contrasts and source fingerprints |
| [vqa_summary.csv](../results/vqa_summary.csv) | Full-budget original VQA aggregate results |
| [lesion_summary.csv](../results/lesion_summary.csv) | Aggregate four-lesion confirmation results |
| [primary_comparisons.csv](../results/primary_comparisons.csv) | Paired group-level differences and 95% intervals |
| [export_checks.json](../results/export_checks.json) | Actual release-export checks and original independent-review fingerprints |

## Checks actually completed in the research archive

- Original VQA: independent final analysis covered **230 prediction files**, **96 original-image groups** and **192 canonical questions**. Full-budget B1/M1 seed-17 checkpoints were reloaded for all 192 clean questions, reproducing candidate scores and recorded generation/parse outcomes exactly.
- Lesion follow-up: all **14 primary tasks** and **4 fresh reproduction tasks** completed. Independent numerical review compared **10,573 scalar values** with no discrepancy. Independent row review checked the **1,575 model-image** lesion rows and the staged question records.
- The extension's reference review independently decoded all **900 official lesion-mask channels** for the 225 images and checked original-image identities. This verifies implementation against the provided annotations; it is not an independent ophthalmologist review.
- Public export: means were independently recomputed from the reviewed original VQA run records; confusion counts and binary macro-F1 were independently recomputed from all **1,575 lesion model-image rows**. All seven existing BLIP-2 variants matched the archived lesion results.

The original archive contains restricted image-level records, model checkpoints, role receipts, failures, and full research reports. They are **not distributed** here. Their hashes provide provenance, not public access to protected assets or a claim that a hash alone independently proves experimental correctness.

## Reproduction boundary

The public repository supplies portable code, configuration and de-identified aggregate statistics. Local figure regeneration and consistency checks can run without medical data. New training and evaluation require your own authorized data and model access.

The original private checkpoint and identity/label records are not downloadable from this repository, so exact historical per-image replay is not advertised as a one-command public benchmark. A new run can reproduce the method under a legally acquired dataset and declared split; it must report its own identity audit and results.

The release does not overwrite the original research archive or erase its **I024** historical label-decoding disclosure. See [study design](study.md#historical-audit-note).
