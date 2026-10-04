# Dataset card

## Source

This project uses the original full CLINC150 dataset from the authors' public repository, pinned to commit `828f8093932c8fe6ca7936c3d2e52903b1c523de`. The included `data/clinc_oos_full.json` is the unchanged 2.50 MB source file. Its SHA-256 is `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`; the program checks that hash before reading it.

The dataset was developed for research on intent classification and out-of-scope queries. Its source describes English queries written by crowdsourced workers in response to scenarios or by paraphrasing seed phrases. It does not contain hospital records or clinical advice.

## Splits and labels

CLINC150 provides 150 supported intents, with 100 examples per intent in training, 20 in validation, and 30 in test. Separate `oos` examples are supplied: 100 for training, 100 for validation, and 1,000 for test. Training combines the 15,000 intent examples with the 100 training `oos` examples. Validation and test combine only their matching in-scope and `oos` splits. The test rows are not used for fitting or checkpoint selection.

The classifier has 151 output classes: 150 intent names and one `oos` label. It predicts one label for every message. Treating `oos` as a label is not the same as having a calibrated confidence threshold, an abstention policy, or a safe request-escalation workflow.

## Evaluation limits

- The 150 examples per intent used for training/validation are short English queries written for a task-oriented dialogue benchmark.
- The test set has a different `oos` proportion from the combined training or validation set; report the published test mix and avoid transferring the resulting precision/recall to another deployment population.
- The reported comparison is one run at a fixed random seed; it includes no repeated-seed variation, confidence interval, or statistical-significance analysis.
- The benchmark covers its defined intents, not the many ways real people describe requests to a hospital.
- The source has exact utterances shared across split boundaries: three between train and validation (two have different labels), and two between train and test (both have different labels). The project preserves the published splits and counts these overlaps in the report; the test scores inherit this source limitation.
- A high benchmark score would not establish clinical safety, deployment readiness, or general performance on healthcare documents.

This repository distributes the original dataset under its published CC BY 3.0 terms, with attribution in [data/DATASET_NOTICE.md](data/DATASET_NOTICE.md). Project code is separately licensed under MIT. The pretrained model has its own Apache 2.0 license.
