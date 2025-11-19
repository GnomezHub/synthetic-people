#!/usr/bin/env python3
import json
import pandas as pd

import sys

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_metrics(gold_data, pred_data):
    """
    Compare gold vs predicted entities and compute per-label and overall metrics.
    """
    
    print(len(gold_data))
    print(len(pred_data))

    # Flatten all entities with their doc id
    gold_entities = []
    pred_entities = []

    for g, p in zip(gold_data, pred_data):
        doc_id = g["id"]

        for ent in g.get("gold_entities", []):
            gold_entities.append({
                "doc_id": doc_id,
                "label": ent["label"],
                "start": ent["start"],
                "end": ent["end"],
                "text": ent["text"]
            })

        for ent in p.get("predicted_entities", []):
            pred_entities.append({
                "doc_id": doc_id,
                "label": ent["label"],
                "start": ent["start"],
                "end": ent["end"],
                "text": ent["text"]
            })

    # Convert to DataFrames
    columns = ["doc_id","label","start","end","text"]
    
    df_gold = pd.DataFrame(gold_entities, columns=columns)
    df_pred = pd.DataFrame(pred_entities, columns=columns)

    # Use exact match: label + start + end + text
    matched = pd.merge(df_pred, df_gold, how="inner", on=["doc_id","label","start","end","text"])
    true_positives = len(matched)
    total_pred = len(df_pred)
    total_gold = len(df_gold)

    precision = true_positives / total_pred if total_pred > 0 else 0.0
    recall = true_positives / total_gold if total_gold > 0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0

    print("OVERALL METRICS:")
    print(f"  Gold entities: {total_gold}")
    print(f"  Predicted entities: {total_pred}")
    print(f"  True positives: {true_positives}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}\n")

    # Per-label metrics
    labels = set(df_gold['label'].unique()).union(df_pred['label'].unique())
    results = []

    for label in sorted(labels):
        df_gold_label = df_gold[df_gold['label'] == label]
        df_pred_label = df_pred[df_pred['label'] == label]
        matched_label = pd.merge(df_pred_label, df_gold_label, how="inner", on=["doc_id","label","start","end","text"])

        tp = len(matched_label)
        total_p = len(df_pred_label)
        total_g = len(df_gold_label)
        precision_l = tp / total_p if total_p > 0 else 0.0
        recall_l = tp / total_g if total_g > 0 else 0.0
        f1_l = 2*precision_l*recall_l/(precision_l+recall_l) if (precision_l+recall_l) > 0 else 0.0

        results.append({
            "label": label,
            "gold_count": total_g,
            "pred_count": total_p,
            "true_positives": tp,
            "precision": precision_l,
            "recall": recall_l,
            "f1": f1_l
        })

    df_results = pd.DataFrame(results)
    print("PER-LABEL METRICS:")
    print(df_results.to_string(index=False))

    return df_results

def main(gold_file, pred_file, save_csv=None):
    gold_data = load_json(gold_file)
    pred_data = load_json(pred_file)

    df_metrics = compute_metrics(gold_data, pred_data)

    if save_csv:
        df_metrics.to_csv(save_csv, index=False)
        print(f"\nMetrics saved to {save_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python eval_ner.py gold.json predictions.json [output_metrics.csv]")
        sys.exit(1)

    gold_file = sys.argv[1]
    pred_file = sys.argv[2]
    save_csv = sys.argv[3] if len(sys.argv) > 3 else None

    main(gold_file, pred_file, save_csv)
