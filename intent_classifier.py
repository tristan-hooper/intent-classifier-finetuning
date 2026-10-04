"""Fine-tune a small pretrained BERT model for intent classification."""

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from itertools import combinations
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import assert_all_finite
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "project_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
DATA_PATH = ROOT / CONFIG["dataset_file"]
DEFAULT_MODEL_DIR = ROOT / "models" / "clinc-bert-tiny"
DEFAULT_REPORT_DIR = ROOT / "reports"
SPLIT_SIZES = {
    "train": 15000, "val": 3000, "test": 4500,
    "oos_train": 100, "oos_val": 100, "oos_test": 1000,
}
BASELINE_CONFIG = {
    "features": "word-level TF-IDF",
    "ngram_range": [1, 2],
    "sublinear_tf": True,
    "classifier": "LogisticRegression",
    "solver": "saga",
    "max_iter": 100,
    "tolerance": 0.001,
    "seed": CONFIG["seed"],
}


def load_splits(path=DATA_PATH):
    """Check the pinned original dataset and return explicit train/validation/test rows."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG["dataset_sha256"]:
        raise ValueError("Dataset SHA-256 does not match project_config.json")
    source = json.loads(raw)
    if set(source) != set(SPLIT_SIZES):
        raise ValueError("Dataset split names do not match the expected full CLINC150 release")
    for name, size in SPLIT_SIZES.items():
        rows = source[name]
        if not isinstance(rows, list) or len(rows) != size:
            raise ValueError(f"Unexpected row count in split {name}")
        if any(
            not isinstance(row, list) or len(row) != 2
            or any(not isinstance(value, str) or not value.strip() for value in row)
            for row in rows
        ):
            raise ValueError(f"Each row in {name} must contain nonblank text and label strings")

    train = source["train"] + source["oos_train"]
    validation = source["val"] + source["oos_val"]
    test = source["test"] + source["oos_test"]
    class_names = sorted({label for _, label in train})
    if len(class_names) != 151 or "oos" not in class_names:
        raise ValueError("Training split must contain 150 intents plus the oos class")
    label_to_id = {label: index for index, label in enumerate(class_names)}
    expected_training_counts = Counter(label for _, label in train)
    if set(expected_training_counts.values()) != {100}:
        raise ValueError("Expected exactly 100 training examples per class, including oos")
    for name, rows in (("validation", validation), ("test", test)):
        unexpected = {label for _, label in rows} - set(label_to_id)
        if unexpected:
            raise ValueError(f"{name} contains labels absent from training: {sorted(unexpected)}")
    if sum(label == "oos" for _, label in validation) != 100:
        raise ValueError("Validation split must contain exactly 100 oos examples")
    if sum(label == "oos" for _, label in test) != 1000:
        raise ValueError("Test split must contain exactly 1,000 oos examples")
    return train, validation, test, class_names, split_overlap_audit(source)


def split_overlap_audit(source):
    """Count exact utterance overlaps and label disagreements across original splits."""
    result = {}
    split_order = {name: index for index, name in enumerate(SPLIT_SIZES)}
    for first, second in combinations(source, 2):
        left, right = sorted((first, second), key=split_order.__getitem__)
        left_labels = {}
        right_labels = {}
        for text, label in source[left]:
            left_labels.setdefault(text, set()).add(label)
        for text, label in source[right]:
            right_labels.setdefault(text, set()).add(label)
        shared = set(left_labels) & set(right_labels)
        disagreements = sum(left_labels[text] != right_labels[text] for text in shared)
        result[f"{left}__{right}"] = {
            "shared_utterances": len(shared),
            "label_disagreements": disagreements,
        }
    return result


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))


def encode_rows(rows, tokenizer, label_to_id, max_length):
    texts = [text for text, _ in rows]
    labels = [label_to_id[label] for _, label in rows]
    untruncated = tokenizer(texts, truncation=False, padding=False)["input_ids"]
    longest = max(map(len, untruncated))
    if longest > max_length:
        raise ValueError(
            f"A dataset query has {longest} wordpieces, exceeding max_length={max_length}; "
            "increase the configured length instead of silently truncating it."
        )
    encoded = tokenizer(
        texts, padding="max_length", truncation=False,
        max_length=max_length, return_tensors="pt",
    )
    dataset = TensorDataset(
        encoded["input_ids"], encoded["attention_mask"], torch.tensor(labels, dtype=torch.long),
    )
    return dataset, longest


def predict_batches(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_predictions = []
    all_confidences = []
    all_truth = []
    loss_sum = 0.0
    with torch.inference_mode():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            probabilities = torch.softmax(logits, dim=-1)
            confidences, predictions = probabilities.max(dim=-1)
            all_predictions.extend(predictions.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())
            all_truth.extend(labels.cpu().tolist())
    assert_all_finite(
        np.asarray(all_confidences), input_name="Model confidence",
    )
    return np.asarray(all_truth), np.asarray(all_predictions), np.asarray(all_confidences), loss_sum / len(dataset)


def score_predictions(truth, predictions, class_names):
    truth = np.asarray(truth, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    label_ids = np.arange(len(class_names))
    oos_id = class_names.index("oos")
    oos_p, oos_r, oos_f, _ = precision_recall_fscore_support(
        truth, predictions, labels=[oos_id], average=None, zero_division=0,
    )
    known_mask = truth != oos_id
    return {
        "examples": int(len(truth)),
        "class_count_including_oos": int(len(class_names)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1_all_classes": float(f1_score(
            truth, predictions, labels=label_ids, average="macro", zero_division=0,
        )),
        "known_intent_accuracy": float(accuracy_score(truth[known_mask], predictions[known_mask])),
        "oos_precision": float(oos_p[0]),
        "oos_recall": float(oos_r[0]),
        "oos_f1": float(oos_f[0]),
        "oos_examples": int((truth == oos_id).sum()),
        "predicted_oos_examples": int((predictions == oos_id).sum()),
        "oos_confusion_matrix": confusion_matrix(
            truth == oos_id, predictions == oos_id, labels=[False, True],
        ).tolist(),
    }


def fit_baseline(train_rows, val_rows, class_names):
    """Fit a lexical baseline on training text and use validation only for review."""
    encoder = LabelEncoder().fit(class_names)
    model = make_pipeline(
        TfidfVectorizer(
            ngram_range=tuple(BASELINE_CONFIG["ngram_range"]),
            sublinear_tf=BASELINE_CONFIG["sublinear_tf"],
        ),
        LogisticRegression(
            solver=BASELINE_CONFIG["solver"], max_iter=BASELINE_CONFIG["max_iter"],
            tol=BASELINE_CONFIG["tolerance"], random_state=BASELINE_CONFIG["seed"],
        ),
    )
    model.fit([text for text, _ in train_rows], [label for _, label in train_rows])
    val_predictions = model.predict([text for text, _ in val_rows])
    return model, {
        "validation": score_predictions(
            encoder.transform([label for _, label in val_rows]),
            encoder.transform(val_predictions), class_names,
        ),
        "label_encoder": encoder,
    }


def make_transformer(model_id, revision, class_names):
    label_to_id = {name: index for index, name in enumerate(class_names)}
    return AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, num_labels=len(class_names),
        id2label={index: name for name, index in label_to_id.items()},
        label2id=label_to_id, cache_dir=str(ROOT / ".cache" / "huggingface"),
    )


def fit_transformer(train_rows, val_rows, test_rows, class_names, output_dir):
    seed_everything(CONFIG["seed"])
    model_id, revision = CONFIG["model_id"], CONFIG["model_revision"]
    cache_dir = str(ROOT / ".cache" / "huggingface")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, use_fast=True, cache_dir=cache_dir,
    )
    label_to_id = {name: index for index, name in enumerate(class_names)}
    train_set, train_longest = encode_rows(train_rows, tokenizer, label_to_id, CONFIG["max_length"])
    val_set, val_longest = encode_rows(val_rows, tokenizer, label_to_id, CONFIG["max_length"])
    test_set, test_longest = encode_rows(test_rows, tokenizer, label_to_id, CONFIG["max_length"])
    model = make_transformer(model_id, revision, class_names)
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    generator = torch.Generator().manual_seed(CONFIG["seed"])
    train_loader = DataLoader(
        train_set, batch_size=CONFIG["batch_size"], shuffle=True,
        generator=generator, num_workers=0,
    )
    history = []
    best_validation_macro_f1 = -1.0
    best_validation_metrics = None
    best_epoch = None
    best_dir = Path(output_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    for epoch in range(1, CONFIG["epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_total = 0.0
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = (
                input_ids.to(device), attention_mask.to(device), labels.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            if not torch.isfinite(loss):
                raise ValueError("Training loss became non-finite")
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(labels)

        truth, predictions, _, val_loss = predict_batches(
            model, val_set, device, CONFIG["batch_size"],
        )
        val_metrics = score_predictions(truth, predictions, class_names)
        row = {
            "epoch": epoch,
            "training_loss": loss_total / len(train_set),
            "validation_loss": val_loss,
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1_all_classes"],
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["validation_macro_f1"] > best_validation_macro_f1:
            best_validation_macro_f1 = row["validation_macro_f1"]
            best_validation_metrics = val_metrics
            best_epoch = epoch
            model.config.id2label = {index: name for index, name in enumerate(class_names)}
            model.config.label2id = label_to_id
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)

    # Re-load the validation-selected checkpoint and evaluate the test split exactly once.
    model = AutoModelForSequenceClassification.from_pretrained(best_dir, local_files_only=True)
    model.to(device)
    test_truth, test_predictions, confidences, test_loss = predict_batches(
        model, test_set, device, CONFIG["batch_size"],
    )
    return {
        "selection_metric": "validation macro F1 over 151 classes",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_validation_macro_f1,
        "validation": best_validation_metrics,
        "test_loss": test_loss,
        "test": score_predictions(test_truth, test_predictions, class_names),
        "predictions": test_predictions.tolist(),
        "max_softmax_confidence": confidences.tolist(),
        "training_seconds": time.perf_counter() - started,
        "max_input_wordpieces": {
            "train": train_longest, "validation": val_longest, "test": test_longest,
        },
        "epochs": history,
    }


def top_confusions(true_ids, predicted_ids, class_names, count=10):
    oos_id = class_names.index("oos")
    confusions = Counter(
        (class_names[true], class_names[predicted])
        for true, predicted in zip(true_ids, predicted_ids)
        if true != predicted and true != oos_id and predicted != oos_id
    )
    return [
        {"true_intent": true, "predicted_intent": predicted, "count": n}
        for (true, predicted), n in confusions.most_common(count)
    ]


def write_reports(
    report_dir, class_names, test_rows, tfidf_result, transformer_result,
    environment, overlap_audit,
):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    label_to_id = {name: index for index, name in enumerate(class_names)}
    truth = [label_to_id[label] for _, label in test_rows]
    content = {
        "dataset": {
            "id": CONFIG["dataset_id"], "revision": CONFIG["dataset_revision"],
            "sha256": CONFIG["dataset_sha256"], "test_examples": len(test_rows),
            "train_examples": SPLIT_SIZES["train"] + SPLIT_SIZES["oos_train"],
            "validation_examples": SPLIT_SIZES["val"] + SPLIT_SIZES["oos_val"],
            "intents": len(class_names) - 1, "out_of_scope_class": "oos",
            "cross_split_exact_text_overlaps": overlap_audit,
        },
        "model": {
            "id": CONFIG["model_id"], "revision": CONFIG["model_revision"],
            "selection_metric": transformer_result["selection_metric"],
            "best_epoch": transformer_result["best_epoch"],
            "best_validation_macro_f1": transformer_result["best_validation_macro_f1"],
            "test_loss": transformer_result["test_loss"],
            "max_input_wordpieces": transformer_result["max_input_wordpieces"],
            "epochs": transformer_result["epochs"],
            "training_seconds": transformer_result["training_seconds"],
        },
        "run_config": {
            "seed": CONFIG["seed"],
            "max_wordpieces": CONFIG["max_length"],
            "batch_size": CONFIG["batch_size"],
            "maximum_epochs": CONFIG["epochs"],
            "learning_rate": CONFIG["learning_rate"],
            "optimizer": "AdamW",
            "device": "cpu",
            "checkpoint_selection": "highest validation macro F1 over 151 classes",
            "baseline": BASELINE_CONFIG,
        },
        "environment": environment,
        "metrics": {
            "tfidf_logistic_regression": tfidf_result["test"],
            "fine_tuned_bert_tiny": transformer_result["test"],
        },
        "validation_metrics": {
            "tfidf_logistic_regression": tfidf_result["validation"],
            "fine_tuned_bert_tiny": transformer_result["validation"],
        },
        "fit_and_evaluation_seconds": {
            "tfidf_fit_and_validation": tfidf_result["seconds"],
            "bert_training_and_test": transformer_result["training_seconds"],
        },
        "top_known_intent_confusions": {
            "tfidf_logistic_regression": top_confusions(
                truth, [label_to_id[label] for label in tfidf_result["predictions"]], class_names,
            ),
            "fine_tuned_bert_tiny": top_confusions(
                truth, transformer_result["predictions"], class_names,
            ),
        },
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    with (report_dir / "test_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for index, (text, label) in enumerate(test_rows):
            for method, predicted, confidence in (
                ("tfidf_logistic_regression", tfidf_result["predictions"][index], None),
                ("fine_tuned_bert_tiny", class_names[transformer_result["predictions"][index]],
                 transformer_result["max_softmax_confidence"][index]),
            ):
                file.write(json.dumps({
                    "method": method, "text": text, "true_label": label,
                    "predicted_label": predicted,
                    "max_softmax_probability": confidence,
                }, ensure_ascii=False) + "\n")

    methods = content["metrics"]
    rows = []
    for method, metric in methods.items():
        rows.append(
            f"| {method} | {metric['examples']} | {metric['accuracy']:.2%} | "
            f"{metric['macro_f1_all_classes']:.4f} | {metric['known_intent_accuracy']:.2%} | "
            f"{metric['oos_precision']:.2%} | {metric['oos_recall']:.2%} | {metric['oos_f1']:.4f} |"
        )
    lines = [
        "# Intent classification evaluation", "",
        "Both methods were fit on the same training examples. The transformer checkpoint",
        "was selected by validation macro F1. The test set was held aside until selection.", "",
        f"The fixed test split contains {len(test_rows)} requests: "
        f"{len(test_rows) - 1000} in-scope examples across 150 intents and 1,000 out-of-scope examples.", "",
        "| Method | Test examples | Accuracy | Macro F1 (151 classes) | Known-intent accuracy | OOS precision | OOS recall | OOS F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows, "",
        "Macro F1 gives each class equal weight. Known-intent accuracy considers only",
        "the 4,500 in-scope test examples. OOS precision and recall treat `oos` as its",
        "own learned class. The model predicts one of 150 intents or `oos`; it has no",
        "confidence threshold or abstention policy. Softmax scores are not calibrated",
        "confidence probabilities.", "",
        "The published test split contains a larger share of OOS examples than training",
        "and validation. Report OOS results in the context of that fixed benchmark mix.", "",
        f"Source overlap audit: {overlap_audit['train__val']['shared_utterances']} exact "
        f"train/validation utterances ({overlap_audit['train__val']['label_disagreements']} "
        "with conflicting labels), and "
        f"{overlap_audit['train__test']['shared_utterances']} exact train/test utterances "
        f"({overlap_audit['train__test']['label_disagreements']} with conflicting labels).",
        "The published splits are preserved; these overlaps are a limitation of the source data.", "",
        "## Fine-tuning history", "",
        "| Epoch | Train loss | Validation loss | Validation accuracy | Validation macro F1 | Seconds |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for epoch in transformer_result["epochs"]:
        lines.append(
            f"| {epoch['epoch']} | {epoch['training_loss']:.4f} | {epoch['validation_loss']:.4f} | "
            f"{epoch['validation_accuracy']:.2%} | {epoch['validation_macro_f1']:.4f} | "
            f"{epoch['seconds']:.1f} |"
        )
    lines += ["", "## Most frequent known-intent confusions", ""]
    for method, confusions in content["top_known_intent_confusions"].items():
        lines += [f"### {method}", "", "| Labeled intent | Predicted intent | Test examples |", "| --- | --- | ---: |"]
        lines.extend(
            f"| {item['true_intent']} | {item['predicted_intent']} | {item['count']} |"
            for item in confusions
        )
        if not confusions:
            lines.append("No known-intent errors.")
        lines.append("")
    lines += [
        "The [metrics file](metrics.json) records pinned input/model revisions and",
        "environment versions. [Every test prediction](test_predictions.jsonl) is",
        "included for inspection. Results measure only this benchmark, not hospital",
        "messages or production performance.",
    ]
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    tick_labels = ["Known intent", "OOS"]
    for ax, (method, metric) in zip(axes, methods.items()):
        matrix = np.asarray(metric["oos_confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], tick_labels, rotation=20, ha="right")
        ax.set_yticks([0, 1], tick_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(method.replace("_", " "))
        for (row, column), value in np.ndenumerate(matrix):
            ax.text(column, row, str(value), ha="center", va="center")
    fig.suptitle("Known intent vs out-of-scope test results")
    fig.tight_layout()
    fig.savefig(report_dir / "oos_confusion.png", dpi=160)
    plt.close(fig)
    return content


def train_project(model_dir=DEFAULT_MODEL_DIR, report_dir=DEFAULT_REPORT_DIR):
    train, validation, test, class_names, overlap_audit = load_splits()
    baseline_start = time.perf_counter()
    baseline_model, baseline = fit_baseline(train, validation, class_names)
    baseline["seconds"] = time.perf_counter() - baseline_start
    model_result = fit_transformer(train, validation, test, class_names, model_dir)
    # Do not score any test examples until the transformer checkpoint is selected.
    baseline_predictions = baseline_model.predict([text for text, _ in test])
    label_encoder = baseline["label_encoder"]
    baseline["predictions"] = baseline_predictions.tolist()
    baseline["test"] = score_predictions(
        label_encoder.transform([label for _, label in test]),
        label_encoder.transform(baseline_predictions), class_names,
    )
    environment = {
        "python": __import__("platform").python_version(),
        **{package: version(package) for package in (
            "numpy", "scikit-learn", "torch", "transformers", "huggingface-hub",
        )},
        "device": "cpu",
    }
    content = write_reports(
        report_dir, class_names, test, baseline, model_result, environment, overlap_audit,
    )
    print(json.dumps({
        "baseline_seconds": baseline["seconds"],
        "training_seconds": model_result["training_seconds"],
        "test_metrics": content["metrics"],
        "reports": str(report_dir),
        "model_checkpoint": str(model_dir),
    }, indent=2))
    return content


def load_classifier(model_dir):
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError("Trained checkpoint not found. Run the train command first.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    return tokenizer, model


def predict(text, model_dir=DEFAULT_MODEL_DIR):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a nonblank string")
    tokenizer, model = load_classifier(model_dir)
    encoded = tokenizer(text, truncation=False, return_tensors="pt")
    if encoded["input_ids"].shape[1] > CONFIG["max_length"]:
        raise ValueError(f"Text exceeds the configured {CONFIG['max_length']}-wordpiece input limit")
    with torch.inference_mode():
        logits = model(**encoded).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        confidence, index = probabilities.max(dim=-1)
    id_to_label = {int(key): value for key, value in model.config.id2label.items()}
    return {
        "intent": id_to_label[index.item()],
        "max_softmax_probability": float(confidence),
        "probability_is_calibrated_confidence": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Fit the baseline and fine-tune BERT-Tiny")
    train_parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    train_parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    prediction = commands.add_parser("predict", help="Predict one request with a saved checkpoint")
    prediction.add_argument("text")
    prediction.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    try:
        if args.command == "train":
            train_project(args.model_dir, args.report_dir)
        else:
            print(json.dumps(predict(args.text, args.model_dir), indent=2))
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
