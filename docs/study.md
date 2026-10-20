# Study design

[Overview](../README.md) · [Data](data.md) · [Results](results.md) · [Evidence](evidence.md) · [Reproduction](reproducibility.md)

## Research question

Can adapting the bridge between a pretrained image encoder and a pretrained language model improve **closed-answer retinal visual question answering**? A follow-up tests whether the same existing adapters recognize independently annotated retinal lesions on a separate dataset.

This project is an adaptation and evaluation study built on [BLIP-2](https://proceedings.mlr.press/v202/li23q.html), not a claim to a new foundation-model architecture.

## What changes in the model

```text
Retinal image → frozen image encoder → Q-Former/query tokens → projection → frozen OPT → answer score
                                            train in M1       train B1/M1
```

| Model | Trainable components | Role |
|---|---|---|
| B0 | None | Original pretrained model |
| B1 | Language projection | Isolate projection adaptation |
| M1 | Q-Former, learned query tokens, language projection | Joint adaptation of the visual-language bridge |
| Simple visual control | Lightweight question-conditioned classifier on frozen visual features | Task-difficulty control |

The image encoder and language-model **parameters** remain frozen. The frozen language-model forward pass still retains the graph needed to transmit gradients to the visual prefix; freezing parameters is different from wrapping that path in `no_grad()`. The frozen image encoder can be cached under fixed preprocessing. Frozen modules stay in evaluation mode while the adapters train.

Training supervises answer tokens only; question/prefix/padding tokens do not contribute to the answer loss. Evaluation scores a fixed candidate set by answer-token conditional likelihood. These results describe closed-answer scoring, not unrestricted clinical text generation.

## Two cohorts, two endpoints

| | Original VQA study | Lesion-presence follow-up |
|---|---|---|
| Reference | Eligible whole-image DME-VQA questions | Official DDR lesion masks |
| Development | 304 training + 77 validation original-image groups | 64 training + 32 validation images for protocol/probe development |
| Evaluation | 96 original-image groups, 192 questions | 225 images, four lesion questions each |
| Endpoint | Type-balanced original-group answer accuracy | Mean binary macro-F1 across four lesion types |
| Adapted models | Three fixed training seeds, 17 / 29 / 43 | The same existing full-budget adapters |

The lesion follow-up does **not** retrain the adapters on the 225-image evaluation cohort. Its development set is a separate source of probe/selection costs and is not added to the original VQA training budget. Repeated models are not additional patients.

Original P1 also executed nested 10%, 25%, and 100% training-budget experiments and question-prior controls. The public highlights use the prespecified full-budget VQA comparison and the separately frozen lesion endpoint; they do not establish a lower total annotation cost.

## Evaluation discipline

- Original-image identities and known duplicate groups are assigned before question derivation; QA rows are not randomly split.
- The follow-up fixes models, candidate answers, parsing, preprocessing, metrics and statistical rules before opening its new confirmation data.
- Paired bootstrap resampling uses original-image groups and applies the same sampled groups to all three fixed seeds before averaging. Seed standard deviation and sample confidence intervals are distinct.
- Independent implementations recompute aggregate results and check raw output records. Fixed checkpoints are also loaded in fresh processes for actual inference reproduction.

## Scope of the public claims

The supported claims concern the **aggregate closed-answer VQA endpoint** and **aggregate reference lesion-presence F1**. They do not establish lesion localization, segmentation, uniform improvement on each lesion, all-lesion correctness, causal grounding, clinical reasoning, free-text factuality, clinical deployment, or SOTA performance.

The full research archive also contains matched representation probes and staged evidence/rule diagnostics. These are distinct from the two public headline endpoints and are not used to imply a clinical-reasoning improvement. Complete same-image clinical evidence plus an independently verified medical answer remains a prerequisite for that stronger research question.

## Historical audit note

**I024:** an early original-study importer decoded raw test-label records before filtering by data role. The audit found no use of these records for fitting or model selection, but the stronger claim that those labels had never been read was withdrawn. Later role-guarded pipelines do not erase this history. The original VQA cohort is not presented as an entirely unread historical test set.

Patient identifiers were unavailable. The auditable split unit is the verified original image/known duplicate group; neither patient-disjoint evaluation nor complete pretraining-data independence is claimed. Exact overlap checks do not prove the absence of every near duplicate.
