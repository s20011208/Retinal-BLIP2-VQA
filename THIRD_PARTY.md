# Third-party materials and permissions

Checked against the source study records and official releases on **2026-10-20**. Software, model and data permissions are separate. The repository's [MIT LICENSE](LICENSE), selected by the maintainer, covers its original code, documentation and schematic artwork. It does not relicense third-party materials, pretrained weights or datasets.

## Pretrained models

| Material | Reference used in the study | Terms and release scope |
|---|---|---|
| Salesforce BLIP-2 OPT-2.7B | [`Salesforce/blip2-opt-2.7b`](https://huggingface.co/Salesforce/blip2-opt-2.7b), revision `59a1ef6c1e5117b3f65523d1c6066825bcf315e3` | Model card declares MIT; embedded model terms remain separate. No weights are redistributed. |
| Embedded Meta OPT-2.7B | [`facebook/opt-2.7b`](https://huggingface.co/facebook/opt-2.7b), source-study reference revision `905a4b602cda5c501f1b3a2650a4152680238254` | [OPT Model License](https://github.com/facebookresearch/metaseq/blob/f7ffa5fd61cf90f498a36d365c13dd7f1a912ff7/projects/OPT/MODEL_LICENSE.md) grants non-commercial research rights subject to its restrictions. BLIP-2's model-card label does not override it. No OPT files are vendored. |

Use the original model distribution and read its full current terms. This code release is not a commercial or clinical model license.

## Datasets

| Source | Authoritative route | Permission boundary |
|---|---|---|
| DME-VQA | [Zenodo 6784358](https://zenodo.org/records/6784358), [author instructions](https://github.com/sergiotasconmorales/consistency_vqa/blob/master/DATA.md) | No usable explicit license was identified in the checked Zenodo license field; original image permissions are separate. No dataset files redistributed. |
| IDRiD | [Official data portal](https://idrid.grand-challenge.org/Data/) | Obtain the authorized source and applicable agreement. Fresh portal access returned 403; no new redistribution grant is inferred. |
| e-ophtha | [ADCIS release and agreement](https://www.adcis.net/en/third-party/e-ophtha/) | Current terms permit non-commercial research/development/teaching and prohibit database sharing. No images, masks or individual labels redistributed. |
| OIA-DDR | [Official author repository](https://github.com/nkicsl/DDR-dataset) | Dataset images, annotations and labels: **CC BY-NC-SA 4.0**. Author repository's code: **MIT**. This repository distributes aggregate statistics, not source data. Any dataset-derived material to which ShareAlike applies retains the applicable CC BY-NC-SA terms. |

The TeleOphta Consortium supplies the e-ophtha databases free of charge ([project site](http://cmm.ensmp.fr/Projects/TeleOphta/index.html)). Follow the attribution requirements in the original agreement when publishing work using them.

## Software dependencies

Dependencies are installed separately and retain their own license files. No package source is vendored. The research runtime used PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121, Transformers 4.46.3, NumPy 1.24.4, SciPy 1.10.1, scikit-learn 1.3.2, Pillow 10.4.0 and Matplotlib 3.7.5. Public installation files specify the supported release environment.

| Dependency | Upstream license family | Official source |
|---|---|---|
| PyTorch / torchvision | BSD-3-Clause; CUDA components separately licensed | [PyTorch](https://github.com/pytorch/pytorch), [torchvision](https://github.com/pytorch/vision) |
| Transformers / Accelerate / Hugging Face Hub / Safetensors | Apache-2.0 | [Hugging Face](https://github.com/huggingface) |
| NumPy / SciPy / scikit-learn | BSD-3-Clause | [NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy), [scikit-learn](https://github.com/scikit-learn/scikit-learn) |
| Pillow | HPND/Pillow license | [Pillow](https://github.com/python-pillow/Pillow) |
| Matplotlib | Matplotlib/PSF-based license | [Matplotlib](https://github.com/matplotlib/matplotlib) |
| PyYAML | MIT | [PyYAML](https://github.com/yaml/pyyaml) |

The public model wrapper calls the installed Transformers API; it does not redistribute the Transformers implementation or pretrained model internals. Diagrams are original schematic artwork; scientific plots are generated from the released aggregate result files and contain no retinal photographs.

## Research references

1. Li J, Li D, Savarese S, Hoi S. **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.** ICML 2023. [Paper](https://proceedings.mlr.press/v202/li23q.html).
2. Tascon-Morales S, Márquez-Neila P, Sznitman R. **Consistency-Preserving Visual Question Answering in Medical Imaging.** MICCAI 2022. [DOI](https://doi.org/10.1007/978-3-031-16452-1_37).
3. Li T, Gao Y, Wang K, Guo S, Liu H, Kang H. **Diagnostic Assessment of Deep Learning Algorithms for Diabetic Retinopathy Screening.** Information Sciences 501, 511–522 (2019). [DOI](https://doi.org/10.1016/j.ins.2019.06.011).
4. Porwal P et al. **Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research.** Data 3(3),25 (2018). [DOI](https://doi.org/10.3390/data3030025).
5. Decencière E et al. **TeleOphta: Machine learning and image processing methods for teleophthalmology.** IRBM (2013). [DOI](https://doi.org/10.1016/j.irbm.2013.01.010).
