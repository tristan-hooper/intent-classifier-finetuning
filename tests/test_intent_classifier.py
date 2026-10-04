import json
from collections import Counter

import pytest
import torch

import intent_classifier as project


def test_pinned_dataset_splits_and_label_inventory():
    train, validation, test, class_names, _ = project.load_splits()
    assert (len(train), len(validation), len(test)) == (15100, 3100, 5500)
    assert len(class_names) == 151
    assert "oos" in class_names
    assert set(Counter(label for _, label in train).values()) == {100}
    assert sum(label == "oos" for _, label in validation) == 100
    assert sum(label == "oos" for _, label in test) == 1000


def test_every_held_out_label_is_in_training_label_map():
    train, validation, test, class_names, _ = project.load_splits()
    train_labels = {label for _, label in train}
    assert set(class_names) == train_labels
    assert {label for _, label in validation} <= train_labels
    assert {label for _, label in test} <= train_labels


def test_source_split_overlap_is_counted_and_reported():
    _, _, _, _, audit = project.load_splits()
    assert audit["train__val"] == {"shared_utterances": 3, "label_disagreements": 2}
    assert audit["train__test"] == {"shared_utterances": 2, "label_disagreements": 2}
    assert audit["val__test"] == {"shared_utterances": 0, "label_disagreements": 0}


def test_changed_source_bytes_are_rejected(tmp_path):
    bad = tmp_path / "changed.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        project.load_splits(bad)


def test_score_metrics_on_hand_computed_example():
    names = ["alarm", "balance", "oos"]
    scores = project.score_predictions(
        truth=[0, 1, 2, 2], predictions=[0, 2, 1, 2], class_names=names,
    )
    assert scores["examples"] == 4
    assert scores["accuracy"] == pytest.approx(0.5)
    assert scores["macro_f1_all_classes"] == pytest.approx(0.5)
    assert scores["known_intent_accuracy"] == pytest.approx(0.5)
    assert scores["oos_precision"] == pytest.approx(0.5)
    assert scores["oos_recall"] == pytest.approx(0.5)
    assert scores["oos_f1"] == pytest.approx(0.5)
    assert scores["oos_confusion_matrix"] == [[1, 1], [1, 1]]


def test_score_rejects_nan_inputs():
    with pytest.raises((ValueError, TypeError)):
        project.score_predictions([0], [float("nan")], ["oos"])


def test_confusions_exclude_correct_predictions_and_oos():
    names = ["account", "balance", "oos"]
    rows = project.top_confusions(
        [0, 0, 1, 2, 2], [1, 1, 0, 0, 2], names,
    )
    assert rows == [
        {"true_intent": "account", "predicted_intent": "balance", "count": 2},
        {"true_intent": "balance", "predicted_intent": "account", "count": 1},
    ]


class FakeTokenizer:
    def __call__(self, texts, truncation, padding, max_length=None, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        token_ids = [[101] + [ord(char) for char in text] + [102] for text in texts]
        if padding == "max_length":
            token_ids = [ids + [0] * (max_length - len(ids)) for ids in token_ids]
            mask = [[int(token != 0) for token in ids] for ids in token_ids]
            return {"input_ids": torch.tensor(token_ids), "attention_mask": torch.tensor(mask)}
        return {"input_ids": token_ids}


def test_encode_rows_fails_instead_of_truncating():
    rows = [("x" * 12, "ok")]
    with pytest.raises(ValueError, match="exceeding max_length"):
        project.encode_rows(rows, FakeTokenizer(), {"ok": 0}, max_length=10)


def test_encode_rows_preserves_labels_and_attention_mask():
    dataset, longest = project.encode_rows(
        [("x", "b"), ("yy", "a")], FakeTokenizer(), {"a": 0, "b": 1}, max_length=5,
    )
    assert longest == 4
    input_ids, mask, labels = dataset.tensors
    assert input_ids.shape == mask.shape == (2, 5)
    assert labels.tolist() == [1, 0]
    assert mask[0].tolist() == [1, 1, 1, 0, 0]


def test_saved_training_evaluation_matches_raw_predictions():
    report = json.loads((project.DEFAULT_REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line) for line in
        (project.DEFAULT_REPORT_DIR / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert report["dataset"]["test_examples"] == 5500
    assert len(rows) == 11000
    class_names = sorted({row["true_label"] for row in rows})
    label_to_id = {label: index for index, label in enumerate(class_names)}
    for method, key in (
        ("tfidf_logistic_regression", "tfidf_logistic_regression"),
        ("fine_tuned_bert_tiny", "fine_tuned_bert_tiny"),
    ):
        selected = [row for row in rows if row["method"] == method]
        assert len(selected) == 5500
        truth = [label_to_id[row["true_label"]] for row in selected]
        predictions = [label_to_id[row["predicted_label"]] for row in selected]
        recomputed = project.score_predictions(truth, predictions, class_names)
        for name, value in recomputed.items():
            if isinstance(value, float):
                assert report["metrics"][key][name] == pytest.approx(value)
            else:
                assert report["metrics"][key][name] == value


def test_finetuning_selection_uses_validation_metric():
    train, validation, test, names, _ = project.load_splits()
    assert len(validation) == 3100 and len(test) == 5500
    assert project.DEFAULT_REPORT_DIR.is_dir()
    report = json.loads((project.DEFAULT_REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert report["model"]["selection_metric"] == "validation macro F1 over 151 classes"
    assert report["model"]["best_epoch"] in range(1, project.CONFIG["epochs"] + 1)
    assert report["run_config"] == {
        "seed": 42,
        "max_wordpieces": 64,
        "batch_size": 32,
        "maximum_epochs": 3,
        "learning_rate": 0.0003,
        "optimizer": "AdamW",
        "device": "cpu",
        "checkpoint_selection": "highest validation macro F1 over 151 classes",
        "baseline": project.BASELINE_CONFIG,
    }
    assert report["dataset"]["train_examples"] == 15100
    assert report["dataset"]["validation_examples"] == 3100
    assert report["dataset"]["cross_split_exact_text_overlaps"]["train__test"] == {
        "shared_utterances": 2, "label_disagreements": 2,
    }


def test_saved_checkpoint_predicts_a_known_label():
    _, _, _, class_names, _ = project.load_splits()
    result = project.predict("I need to check the current balance on my account")
    assert result["intent"] in class_names
    assert 0 <= result["max_softmax_probability"] <= 1
    assert result["probability_is_calibrated_confidence"] is False
