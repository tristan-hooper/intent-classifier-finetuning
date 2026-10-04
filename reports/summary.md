# Intent classification evaluation

Both methods were fit on the same training examples. The transformer checkpoint
was selected by validation macro F1. The test set was held aside until selection.

The fixed test split contains 5500 requests: 4500 in-scope examples across 150 intents and 1,000 out-of-scope examples.

| Method | Test examples | Accuracy | Macro F1 (151 classes) | Known-intent accuracy | OOS precision | OOS recall | OOS F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf_logistic_regression | 5500 | 73.49% | 0.8083 | 88.56% | 85.07% | 5.70% | 0.1068 |
| fine_tuned_bert_tiny | 5500 | 71.45% | 0.7632 | 83.84% | 94.58% | 15.70% | 0.2693 |

Macro F1 gives each class equal weight. Known-intent accuracy considers only
the 4,500 in-scope test examples. OOS precision and recall treat `oos` as its
own learned class. The model predicts one of 150 intents or `oos`; it has no
confidence threshold or abstention policy. Softmax scores are not calibrated
confidence probabilities.

The published test split contains a larger share of OOS examples than training
and validation. Report OOS results in the context of that fixed benchmark mix.

Source overlap audit: 3 exact train/validation utterances (2 with conflicting labels), and 2 exact train/test utterances (2 with conflicting labels).
The published splits are preserved; these overlaps are a limitation of the source data.

## Fine-tuning history

| Epoch | Train loss | Validation loss | Validation accuracy | Validation macro F1 | Seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.7974 | 2.3569 | 63.16% | 0.6072 | 58.2 |
| 2 | 1.5793 | 1.1803 | 74.03% | 0.7304 | 60.6 |
| 3 | 0.7826 | 0.8216 | 81.81% | 0.8201 | 57.8 |

## Most frequent known-intent confusions

### tfidf_logistic_regression

| Labeled intent | Predicted intent | Test examples |
| --- | --- | ---: |
| calendar | calendar_update | 11 |
| change_user_name | user_name | 9 |
| oil_change_how | oil_change_when | 8 |
| redeem_rewards | rewards_balance | 8 |
| meal_suggestion | restaurant_suggestion | 7 |
| pto_request | pto_request_status | 6 |
| improve_credit_score | credit_score | 6 |
| restaurant_suggestion | meal_suggestion | 6 |
| shopping_list_update | shopping_list | 5 |
| transactions | spending_history | 5 |

### fine_tuned_bert_tiny

| Labeled intent | Predicted intent | Test examples |
| --- | --- | ---: |
| oil_change_when | oil_change_how | 30 |
| calendar_update | calendar | 24 |
| what_is_your_name | user_name | 18 |
| redeem_rewards | rewards_balance | 15 |
| distance | directions | 13 |
| shopping_list | shopping_list_update | 13 |
| improve_credit_score | credit_score | 11 |
| change_user_name | user_name | 11 |
| maybe | no | 10 |
| share_location | current_location | 10 |

The [metrics file](metrics.json) records pinned input/model revisions and
environment versions. [Every test prediction](test_predictions.jsonl) is
included for inspection. Results measure only this benchmark, not hospital
messages or production performance.
