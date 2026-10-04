import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
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


@pytest.mark.parametrize(
    ("truth", "predictions"),
    [
        ([0], [0.9]),
        ([0.5], [0]),
        ([0, 1], [0]),
        ([0], [3]),
        ([-1], [0]),
        ([[0]], [0]),
        ([], []),
    ],
)
def test_score_rejects_malformed_class_ids(truth, predictions):
    with pytest.raises(ValueError):
        project.score_predictions(truth, predictions, ["known", "oos", "other"])


def test_score_rejects_invalid_class_inventory():
    with pytest.raises(ValueError, match="unique, nonblank"):
        project.score_predictions([0], [0], ["known", "oos", "oos"])


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


def test_saved_report_records_run_configuration_and_source_limits():
    _, validation, test, _, _ = project.load_splits()
    assert len(validation) == 3100 and len(test) == 5500
    assert project.DEFAULT_REPORT_DIR.is_dir()
    report = json.loads((project.DEFAULT_REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert report["model"]["selection_metric"] == "validation macro F1 over 151 classes"
    assert report["model"]["best_epoch"] in range(1, project.CONFIG["epochs"] + 1)
    assert "training_and_validation_seconds" in report["model"]
    assert "checkpoint_reload_and_test_evaluation_seconds" in report["model"]
    assert "training_seconds" not in report["model"]
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


def test_fit_transformer_saves_checkpoint_with_best_validation_score(monkeypatch, tmp_path):
    class_names = ["alpha", "beta", "oos"]
    train_set = torch.utils.data.TensorDataset(
        torch.tensor([[1, 2]]), torch.tensor([[1, 1]]), torch.tensor([0]),
    )
    validation_set = object()
    test_set = object()
    encoded_sets = iter(((train_set, 2), (validation_set, 2), (test_set, 2)))
    monkeypatch.setattr(project, "encode_rows", lambda *args, **kwargs: next(encoded_sets))
    monkeypatch.setitem(project.CONFIG, "epochs", 3)
    monkeypatch.setitem(project.CONFIG, "batch_size", 1)
    monkeypatch.setitem(project.CONFIG, "max_length", 4)

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def save_pretrained(self, path):
            return None

    saved_epochs = []

    class FakeModel:
        def __init__(self, epoch=0):
            self.epoch = epoch
            self.config = SimpleNamespace()

        def to(self, device):
            return self

        def parameters(self):
            return []

        def train(self):
            self.epoch += 1

        def __call__(self, **kwargs):
            return SimpleNamespace(loss=torch.tensor(1.0, requires_grad=True))

        def save_pretrained(self, path, safe_serialization):
            saved_epochs.append(self.epoch)

    class FakeModelLoader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["local_files_only"] is True
            return FakeModel(saved_epochs[-1])

    class FakeOptimizer:
        def zero_grad(self, set_to_none):
            return None

        def step(self):
            return None

    monkeypatch.setattr(project, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(project, "AutoModelForSequenceClassification", FakeModelLoader)
    monkeypatch.setattr(project, "make_transformer", lambda *args: FakeModel())
    monkeypatch.setattr(project.torch.optim, "AdamW", lambda *args, **kwargs: FakeOptimizer())

    validation_predictions = {
        1: np.asarray([0, 0, 0]),
        2: np.asarray([0, 1, 2]),
        3: np.asarray([0, 0, 0]),
    }
    evaluated = []

    def fake_predict_batches(model, dataset, device, batch_size):
        evaluated.append("validation" if dataset is validation_set else "test")
        truth = np.asarray([0, 1, 2])
        predictions = (
            validation_predictions[model.epoch]
            if dataset is validation_set else np.asarray([0, 1, 2])
        )
        return truth, predictions, np.asarray([0.8, 0.8, 0.8]), 0.5

    monkeypatch.setattr(project, "predict_batches", fake_predict_batches)
    result = project.fit_transformer(
        [("a", "alpha")], [("b", "beta")], [("c", "oos")], class_names, tmp_path / "model",
    )

    assert result["best_epoch"] == 2
    assert saved_epochs == [1, 2]
    assert evaluated == ["validation", "validation", "validation", "test"]
    assert result["test"]["accuracy"] == 1.0


def test_predict_runs_with_a_loaded_checkpoint(monkeypatch):
    class FakeTokenizer:
        def __call__(self, text, truncation, return_tensors):
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

    class FakeModel:
        config = SimpleNamespace(id2label={0: "balance", 1: "oos"})

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[0.0, 2.0]]))

    monkeypatch.setattr(project, "load_classifier", lambda model_dir: (FakeTokenizer(), FakeModel()))
    result = project.predict("check balance")
    assert result["intent"] == "oos"
    assert 0 < result["max_softmax_probability"] <= 1
    assert result["probability_is_calibrated_confidence"] is False


@pytest.mark.integration
def test_saved_checkpoint_predicts_a_known_label():
    if not (project.DEFAULT_MODEL_DIR / "config.json").is_file():
        pytest.skip("Train the model to run this local-checkpoint integration test")
    _, _, _, class_names, _ = project.load_splits()
    result = project.predict("I need to check the current balance on my account")
    assert result["intent"] in class_names
    assert 0 <= result["max_softmax_probability"] <= 1
    assert result["probability_is_calibrated_confidence"] is False
