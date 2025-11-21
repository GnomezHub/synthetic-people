#!/usr/bin/env python3
import json
import pandas as pd
import sys
import os

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_run(gold_data, pred_data, run_id="baseline"):
    gold_entities = []
    pred_entities = []
    detailed_rows = []

    for g, p in zip(gold_data, pred_data):
        doc_id = g["id"]
        text = g["text"]
        gold_list = g.get("gold_entities", [])
        pred_list = p.get("predicted_entities", [])

        # Flatten
        for ent in gold_list:
            gold_entities.append({
                "doc_id": doc_id,
                "label": ent["label"],
                "start": ent["start"],
                "end": ent["end"],
                "text": ent["text"]
            })
        for ent in pred_list:
            pred_entities.append({
                "doc_id": doc_id,
                "label": ent["label"],
                "start": ent["start"],
                "end": ent["end"],
                "text": ent["text"]
            })

        # Detailed row per doc
        gold_set = {(e['label'], e['start'], e['end'], e['text']) for e in gold_list}
        pred_set = {(e['label'], e['start'], e['end'], e['text']) for e in pred_list}
        tp_count = len(gold_set & pred_set)

        detailed_rows.append({
            "run_id": run_id,
            "doc_id": doc_id,
            "text": text,
            "gold_count": len(gold_list),
            "predicted_count": len(pred_list),
            "true_positives": tp_count,
            "notes": ""
        })

    # Convert to DataFrames
    df_gold = pd.DataFrame(gold_entities)
    df_pred = pd.DataFrame(pred_entities)

    # Overall true positives
    matched = pd.merge(df_pred, df_gold, how="inner", on=["doc_id","label","start","end","text"])
    tp = len(matched)
    total_pred = len(df_pred)
    total_gold = len(df_gold)
    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gold if total_gold > 0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0

    # Per-label F1
    labels = set(df_gold['label'].unique()).union(df_pred['label'].unique())
    per_label_f1 = {}
    for label in sorted(labels):
        df_gold_l = df_gold[df_gold['label'] == label]
        df_pred_l = df_pred[df_pred['label'] == label]
        matched_l = pd.merge(df_pred_l, df_gold_l, how="inner", on=["doc_id","label","start","end","text"])
        tp_l = len(matched_l)
        total_p = len(df_pred_l)
        total_g = len(df_gold_l)
        precision_l = tp_l / total_p if total_p > 0 else 0.0
        recall_l = tp_l / total_g if total_g > 0 else 0.0
        f1_l = 2*precision_l*recall_l/(precision_l+recall_l) if (precision_l+recall_l) > 0 else 0.0
        per_label_f1[label] = f1_l

    # Overview row
    overview_row = {
        "run_id": run_id,
        "total_gold": total_gold,
        "total_predicted": total_pred,
        "true_positives": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    for label, f1_val in per_label_f1.items():
        overview_row[f"{label}_f1"] = f1_val

    df_overview = pd.DataFrame([overview_row])
    df_detailed = pd.DataFrame(detailed_rows)

    return df_overview, df_detailed

# def save_or_append(df, filename):
#     if os.path.exists(filename):
#         df.to_csv(filename, mode='a', header=False, index=False)
#     else:
#         df.to_csv(filename, header=False, index=False)
def save(df, filename):
	df.to_csv(filename, header=False, index=False)

def main(gold_file, pred_file, run_id="baseline", overview_file="overview.csv", detailed_file="detailed.csv"):
    gold_data = load_json(gold_file)
    pred_data = load_json(pred_file)

    df_overview, df_detailed = evaluate_run(gold_data, pred_data, run_id=run_id)
    
    save(df_overview, overview_file)
    save(df_detailed, detailed_file)

    print(f"Saved overview metrics to {overview_file}")
    print(f"Saved detailed per-document metrics to {detailed_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python eval_ner.py gold.json predictions.json [run_id]")
        sys.exit(1)
    gold_file = sys.argv[1]
    pred_file = sys.argv[2]
    run_id = sys.argv[3] if len(sys.argv) > 3 else "baseline"
    main(gold_file, pred_file, run_id)