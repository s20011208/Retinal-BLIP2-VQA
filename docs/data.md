# Data and reference labels

[Overview](../README.md) · [Study](study.md) · [Results](results.md) · [Reproduction](reproducibility.md)

This repository contains **no retinal images, individual annotations, per-image predictions, patient identifiers, or trained model weights**. Obtain the data directly from the relevant rights holders and comply with each source's current terms.

## DME-VQA: original whole-image questions

- [Official release: Zenodo 6784358](https://zenodo.org/records/6784358)
- [Author data instructions](https://github.com/sergiotasconmorales/consistency_vqa/blob/master/DATA.md)
- [Original paper](https://doi.org/10.1007/978-3-031-16452-1_37)

The source combines IDRiD and e-ophtha and includes both whole-image and regional template questions. This study uses the audited eligible **whole-image hard-exudate presence** and **dataset-defined DME proxy-grade** endpoints. The proxy label is not an OCT-confirmed diagnosis of clinical macular edema. These related labels are not independent evidence for complex medical reasoning.

The study's eligible counts are 304 training, 77 validation and 96 evaluation original-image groups, with two whole-image questions per group. These are filtered study counts, not the original release's full benchmark counts. Reserved source images and identified overlaps are excluded under the study's data roles.

The Zenodo page does not expose a usable explicit license in the checked public license field. A download link does not supply a blanket redistribution grant. Underlying image rights remain separate:

- [IDRiD official data portal](https://idrid.grand-challenge.org/Data/): obtain the authorized original release and its terms. The portal returned access errors in the fresh publication check; no additional redistribution permission is inferred.
- [e-ophtha / ADCIS](https://www.adcis.net/en/third-party/e-ophtha/): current terms limit use to non-commercial research, development and teaching and prohibit sharing the database. Download requires a personal access application.

## DDR: four independent lesion channels

- [Official author repository and download routes](https://github.com/nkicsl/DDR-dataset)
- [Paper: Diagnostic Assessment of Deep Learning Algorithms for Diabetic Retinopathy Screening](https://doi.org/10.1016/j.ins.2019.06.011)

DDR data use **CC BY-NC-SA 4.0**; the author repository's source code separately uses MIT. Its dataset permission is not replaced by the code license. The public release exports aggregate scientific statistics only.

The follow-up uses 64 development-training, 32 development-validation and 225 confirmation images. Each confirmation image has four existing official lesion-mask channels:

| Code | Reference lesion | Positive masks | Zero-valued masks |
|---|---|---:|---:|
| EX | Hard exudates | 171 | 54 |
| HE | Hemorrhages | 194 | 31 |
| MA | Microaneurysms | 124 | 101 |
| SE | Soft exudates / cotton-wool spots | 42 | 183 |

Presence is derived from whether the corresponding **existing** binary mask has any nonzero pixel. An existing all-zero channel supplies a dataset-reference negative; a missing mask is **unknown**, never an automatic negative. These are reference annotations, not a second independent clinical adjudication of absence.

The evaluated output is image-level presence. A correct yes/no prediction does not establish the lesion's location. A question ROI specifies the region being asked about; it is not lesion segmentation truth and cannot substitute for a lesion mask.

The four presence flags do not encode every clinical severity criterion, its extent or distribution. They therefore cannot certify that every required clinical finding has been correctly recognized before a medical severity answer.

## Prepare your own legal manifest

Use local, configurable paths and preserve original-image grouping. Partition roles before deriving questions or decoding unrelated-role answers. Keep protected labels and image-level records outside the Git repository. See the executable schema and commands in [reproduction](reproducibility.md).

These instructions support a new lawful run. They do not reclassify the already exposed historical evaluation cohorts as new, unseen confirmation data. Newly selected methods require newly held-out confirmation data.
