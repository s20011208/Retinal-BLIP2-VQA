# Reproduction guide

[Overview](../README.md) · [Study](study.md) · [Data permissions](data.md) · [Evidence](evidence.md)

## What runs from this repository

- Regenerate the README's original schematic artwork and scientific charts from the released aggregate JSON.
- Audit a locally authorized whole-image QA manifest without decoding unrelated-role answers.
- Train projection-only B1 or joint-bridge M1, retain complete interruption-recovery state, and evaluate B0/B1/M1 with fixed candidate likelihood and separate strict free-generation parsing.
- Recompute grouped VQA, image-pair and four-lesion metrics from local predictions; compare equally many fixed seed runs with a paired original-group bootstrap.

These are portable implementations of the research's core mechanisms. They are not the original server's full orchestration or a public copy of its restricted test records. The original private adapter checkpoints and image-identity/answer manifests are not distributed; **exact historical per-image replay is not advertised**. New runs require lawful data/model access and must report their own audit and measured results. The released historical aggregates remain unchanged.

## Install

The source research ran Python **3.8.10**, PyTorch **2.4.1+cu121**, Transformers **4.46.3** and Accelerate **0.34.2**. The direct runtime versions are pinned in [requirements-lock.txt](../requirements-lock.txt); this is a direct-dependency pin file, not a complete platform-independent binary lock. Choose the PyTorch wheel matching your own NVIDIA driver/platform; do not replace a shared system environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[dev,figures]'
```

Read the [separate model and data terms](../THIRD_PARTY.md). Obtain the BLIP-2 model through its official distribution and retain the fixed source-study revision:

```text
Salesforce/blip2-opt-2.7b
59a1ef6c1e5117b3f65523d1c6066825bcf315e3
```

The public wrapper defaults to a local model directory. It does not silently download weights or assert an access grant.

## Check the software and regenerate figures

```bash
python -m pytest -q
python tools/build_figures.py
retinal-vqa --help
```

The test fixtures verify mechanisms; they are not retinal experiments or new scientific results. Figure generation uses only [main_results.json](../results/main_results.json) and produces the checked-in SVG/PNG assets plus their source fingerprint manifest.

## Prepare local data

Set separate data, model and output locations. The paths below are local choices, not downloadable data supplied by the repository.

```bash
export RETINAL_DATA_ROOT="$PWD/local-data"
export RETINAL_MODEL_PATH="$PWD/local-models/blip2-opt-2.7b"
export RETINAL_OUTPUT_ROOT="$PWD/local-runs"
```

The loader expects two JSONL files, with one question identity per line. Image paths may be relative to `RETINAL_DATA_ROOT`. Assign original-image/patient groups and roles before deriving QA. A missing patient identifier remains unknown.

**`identities.jsonl` — contains no answers or masks**

| Required field | Meaning |
|---|---|
| `qa_id` | Unique question record identity |
| `canonical_image_id` | Verified original-image identity shared by derived QA |
| `group_id` | Patient group when verified; otherwise original-image/known-duplicate group |
| `role` | Exactly `train`, `val`, or `test` |
| `image_path` | Authorized local whole-image file |
| `source_dataset` | Traceable source name/version |

Optional `patient_id`, `raw_sha256` and `pixel_sha256` support identity checks. The default image audit also computes actual raw-file and decoded-RGB fingerprints and rejects exact content overlap across roles, including different filenames or image encodings. A supplied raw hash must match the file. These checks do not certify patient identity or the absence of every near duplicate. Resolve source permissions, original-image mappings and uncertain group relationships before admitting a study.

**`qa.jsonl` — locally protected reference questions**

| Required field | Meaning |
|---|---|
| `qa_id`, `role` | Must agree with the admitted identity |
| `question`, `question_type` | Frozen text and endpoint |
| `answer_candidates` | Unique nonempty string array, fixed before evaluation |
| `answer` | Reference string belonging to the fixed candidates |
| `answer_provenance` | Verifiable source/derivation |
| `annotation_complete` | Must be `true` for this core |

Ambiguous references and unaudited regional questions are rejected. The core supports whole-image tasks; a question ROI is not accepted as lesion truth. The reader requires canonical literal `role` and `qa_id` markers and skips unrelated roles **before** decoding their answer objects. It refuses duplicate or ambiguous schema keys.

## Audit, train and predict on development data

```bash
retinal-vqa audit \
  --identities "$RETINAL_DATA_ROOT/identities.jsonl" \
  --answers "$RETINAL_DATA_ROOT/qa.jsonl" \
  --data-root "$RETINAL_DATA_ROOT" \
  --roles train val \
  --output "$RETINAL_OUTPUT_ROOT/data-audit.json"

retinal-vqa train --config configs/train_m1.json
retinal-vqa train --config configs/train_b1.json

retinal-vqa predict --config configs/evaluate_b0.json
retinal-vqa predict --config configs/evaluate_m1.json

retinal-vqa score-vqa \
  --predictions "$RETINAL_OUTPUT_ROOT/M1_b100_s17_val.jsonl" \
  --output "$RETINAL_OUTPUT_ROOT/M1_b100_s17_val.metrics-recomputed.json"
```

The training templates preserve the recorded full-budget recipe: AdamW, learning rate `3e-5`, weight decay `0.01`, gradient clip `1.0`, effective batch `16`, at most `20` epochs, validation patience `4`, and checkpoint interval `10` optimizer steps. The source-study configuration records this learning rate as prespecified without a search. The public data format and implementation fingerprint are new.

The supplied configurations evaluate `val` by default. B0 loads no adapter; adapted evaluation requires its own locally trained checkpoint. Use distinct output directories for each method/budget/seed, and align the fixed seeds and training-image budgets across methods. Seeds 17 / 29 / 43 identify training repeats, not additional clinical samples.

Use your server's scheduler or lifecycle GPU lock before starting a worker. This portable CLI does not reserve a shared GPU on your behalf. Keep runs in a persistent supervisor when needed.

Training binds the actual selected-training/validation image bytes and local model weights/config/tokenizer files, alongside the identity/answer manifests, resolved configuration and implementation. A rejected recovery preserves the previous run artifacts; it does not replace them with the attempted input metadata.

Training preserves adapter/optimizer/scheduler/random/sampler states in `checkpoints/latest.pt`; `checkpoints/best.pt` is selected only by validation. Restore the **same** run and resolved configuration with:

```bash
retinal-vqa train --config configs/train_m1.json --resume
```

A new implementation fingerprint is not interchangeable with an interrupted historical training process. Do not load untrusted pickle checkpoints or treat a changed protocol as the same recovery run.

## Score locally produced lesion records

`score-lesions` accepts one record per original image with `sample_id`, `group_id`, four known boolean references in `evidence_truth`, and four `lesions` records containing `prediction` as `"yes"`, `"no"`, or `null` (invalid). The lesion keys are exactly `EX`, `HE`, `MA`, `SE`.

Derive reference presence from an actually existing, identity-matched annotation channel; **never turn a missing mask into `false`**. This scorer does not create medical annotations or automatically assert mask completeness. Use the explicit [DDR question-type mapping](../configs/ddr_lesions.json) to join four locally produced lesion-question predictions by verified original-image identity. This mapping does not create labels or retrieve data. Each original image must have exactly one complete yes/no question for each mapped lesion; unknown or duplicate lesion evidence is rejected.

```bash
retinal-vqa assemble-lesions \
  --predictions "$RETINAL_OUTPUT_ROOT/lesion-qa-predictions.jsonl" \
  --mapping configs/ddr_lesions.json \
  --output "$RETINAL_OUTPUT_ROOT/lesion-predictions.jsonl"

retinal-vqa score-lesions \
  --predictions "$RETINAL_OUTPUT_ROOT/lesion-predictions.jsonl" \
  --output "$RETINAL_OUTPUT_ROOT/lesion-metrics.json" \
  --bootstrap 2000 --seed 20261020
```

The output separates positive precision/recall/F1, binary macro-F1, balanced accuracy, class support, invalid predictions and original-group intervals. Sparse or undefined estimates remain explicitly unsupported rather than being filled with zero.

## Confirmation is a separate gate

Training/audit examples admit development roles only. Test-answer access requires actual image checks and a verified `raw_sha256` for every test identity, plus a local frozen-study receipt that binds the exact identity and answer files; prediction additionally checks the resolved configuration, code fingerprint, actual base-model content and checkpoint. There is deliberately no command that automatically declares a study frozen or authorizes new test use.

Before a new confirmation run, fix data roles, model candidates, parsing, thresholds, metrics and comparisons; obtain the applicable independent review and lawful access. Already exposed original-study or DDR confirmation data cannot become a new unseen test by changing directories or splits. This repository does not recreate or override the original research's access receipts.
