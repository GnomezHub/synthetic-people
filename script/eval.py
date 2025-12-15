import json
import csv
import sys

"""
This script computes two different evaluations. 

The first one only cares about span detection, i.e. correct indexes. Labels are ignored.
This shows the model's ability to correctly indentify sensitive entities. For our task, 
correct indexes are more important than correct label.

The second one is a full NER evaluation, where label counts towards the score. 
Here, both label and indexes must be exactly correct to count as a match. 
F1 is computed for the full dataset as well as for each label, to show which labels the model often gets wrong.

"""

"""
1. Span detection.

True positive = the model predicts a span that matches the gold exactly
False positive = the model predicts a span that is not present in gold
False negative = a gold span that the model missed

Example:

gold_spans = [(10, 15), (20, 25), (30, 40)]
predicted_spans = [(3, 9), (10, 15), (20, 25), (26, 36)]

In this scenario, there are:
    - 2 true positives ((10, 15) and (20, 25))
    - 2 false positives ((3, 9) and (26, 36))
    - 1 false negative ((30, 40))

"""
# Define gold and prediction files and run_id in terminal
if len(sys.argv) < 4:
    print("Usage: python eval.py <run_id> <gold-file> <prediction-file>")
    sys.exit(1)

run_id = sys.argv[1]
gold_file = sys.argv[2]
pred_file = sys.argv[3]

# Read gold file
with open (gold_file, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

# Read prediction file
with open(pred_file, "r", encoding="utf-8") as f:
    pred_data = json.load(f)

# Save all spans in sets for easy comparison later through set operations
gold_spans = set()
pred_spans = set()

# Iterate over all docs to extract all index spans. Add doc ids to avoid collisions if several docs contain the same spans.
for doc in gold_data:
    doc_id = doc["id"]
    for entity in doc["gold_entities"]:
        gold_spans.add((doc_id, entity["start"], entity["end"]))


# Exactly the same for prediction data
for doc in pred_data:
    doc_id = doc["id"]
    for entity in doc["predicted_entities"]:
        pred_spans.add((doc_id, entity["start"], entity["end"]))

# Sets look like: {('sv-02', 190, 215), ('sv-24', 32, 45), ('sv-02', 32, 43), ('sv-14', 11, 21)...}

# Intersection - returns values that are in both sets, i.e. True Positives. 
tp = len(gold_spans & pred_spans)

# Difference - returns values that are in predicted set but not in gold, i.e. False Positives.
fp = len(pred_spans - gold_spans)

# Difference - returns values that are in gold set but not predicted set, i.e. False Negatives.
fn = len(gold_spans - pred_spans)

# Else clause is to avoid division by zero
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

denominator = precision + recall

f1 = 2 * ((precision * recall) / denominator) if denominator > 0 else 0.0

"""
2. Standard NER evaluation

The logic is exactly the same as part one except that here, the predicted label is also taken into account. 

True positive = the model predicts an index span + label that matches the gold exactly
False positive = the model predicts a span + label that are not present in gold
False negative = a gold span + label that the model missed

"""

gold_spans_std = set()
pred_spans_std = set()

for doc in gold_data:
    doc_id = doc["id"]
    for entity in doc["gold_entities"]:
        gold_spans_std.add((doc_id, entity["start"], entity["end"], entity["label"]))

for doc in pred_data:
    doc_id = doc["id"]
    for entity in doc["predicted_entities"]:
        pred_spans_std.add((doc_id, entity["start"], entity["end"], entity["label"]))


tp_std = len(gold_spans_std & pred_spans_std)
fp_std = len(pred_spans_std - gold_spans_std)
fn_std = len(gold_spans_std - pred_spans_std)

precision_std = tp_std / (tp_std + fp_std) if (tp_std + fp_std) > 0 else 0.0
recall_std = tp_std / (tp_std + fn_std) if (tp_std + fn_std) > 0 else 0.0

denominator = precision_std + recall_std
f1_std = 2 * ((precision_std * recall_std) / denominator) if denominator > 0 else 0.0

"""
Compute f1 for each label

True positives = correct predictions for this label
False positives = predictions with this label that are wrong
False negatives = gold entities of this label that the model missed

"""


# Initialize empty dictionaries for each label as a counter
labels = ["NAME", "PHONE", "ADDRESS", "NATIONAL_ID", "EMAIL"]

tp_label = {label: 0 for label in labels}
fp_label = {label: 0 for label in labels}
fn_label = {label: 0 for label in labels}

# Count TP for each label
for item in gold_spans_std & pred_spans_std: # Only spans that match exactly
    label = item[3] # The fourth element of the tuple is the label
    tp_label[label] += 1

# Count FP for each label
for item in pred_spans_std - gold_spans_std: # Predicted but wrong
    label = item[3]
    fp_label[label] += 1

# Count FN for each label
for item in gold_spans_std - pred_spans_std: # Missed spans/labels
    label = item[3]
    fn_label[label] += 1


# Function to compute f1 for a single label
def f1_for_label(tp, fp, fn):
    if tp + fp == 0:
        precision = 0.0

    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0

    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

# Call the function for each label to get a f1 score for each
NAME_f1 = f1_for_label(tp_label["NAME"], fp_label["NAME"], fn_label["NAME"])
PHONE_f1 = f1_for_label(tp_label["PHONE"], fp_label["PHONE"], fn_label["PHONE"])
ADDRESS_f1 = f1_for_label(tp_label["ADDRESS"], fp_label["ADDRESS"], fn_label["ADDRESS"])
NATIONAL_ID_f1 = f1_for_label(tp_label["NATIONAL_ID"], fp_label["NATIONAL_ID"], fn_label["NATIONAL_ID"])
EMAIL_f1 = f1_for_label(tp_label["EMAIL"], fp_label["EMAIL"], fn_label["EMAIL"])

# Finally - save everything to a CSV file

# Define the header
columns = [
    "run_id",
    "eval_type",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
    "NAME_f1",
    "PHONE_f1",
    "ADDRESS_f1",
    "NATIONAL_ID_f1",
    "EMAIL_f1"
]

# Two rows will be produced - one for the span only evaluation and one for the standard evaluation

span_only_row = [
    run_id,
    "span-only",
    precision,
    recall,
    f1,
    tp,
    fp,
    fn
]

standard_row = [
    run_id,
    "standard",
    precision_std,
    recall_std,
    f1_std,
    tp_std,
    fp_std,
    fn_std,
    NAME_f1,
    PHONE_f1,
    ADDRESS_f1,
    NATIONAL_ID_f1,
    EMAIL_f1
]

rows = [span_only_row, standard_row]

with open("metrics.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerows(rows)

print("Done! Result saved to 'metrics.csv'.")