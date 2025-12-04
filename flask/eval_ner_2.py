
import json
import pandas as pd
import sys


# --- Datainläsning och Hjälpfunktioner ---

def load_json_data(file_path):
    """
    Läser in data från en JSON-fil.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Fel: Filen hittades inte på sökvägen: {file_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Fel: Kunde inte tolka JSON från filen: {file_path}", file=sys.stderr)
        sys.exit(1)

def save_dataframe_to_csv(dataframe, file_path):
    """
    Sparar en pandas DataFrame till en CSV-fil, inklusive kolumnrubriker.
    """
    try:

        dataframe.to_csv(file_path, header=False, index=False) # Sätt header =True för att felsöka ifall kolumnerna inte stämmer

    except Exception as e:
        print(f"Fel vid spara DataFrame till {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

# --- Entitetshantering och Normalisering ---

def normalize_entities(document_list, doc_id, source_key):
    """
    Extraherar och normaliserar entiteter från en lista av dokument
    till ett platt format (en entitet per rad i en lista av dictionaries).
    """
    normalized_entities = []
    for entity in document_list:
        normalized_entities.append({
            "doc_id": doc_id,
            "label": entity["label"],
            "start": entity["start"],
            "end": entity["end"],
            "text": entity["text"],
            "source": source_key # 'gold' eller 'prediction'
        })
    return normalized_entities

# --- Beräkning av Metriker ---

def calculate_precision_recall_f1(true_positives, total_predictions, total_gold):
    """
    Beräknar Precision, Recall och F1-score.
    """
    precision = true_positives / total_predictions if total_predictions > 0 else 0.0
    recall = true_positives / total_gold if total_gold > 0 else 0.0
    
    # Förhindra division med noll i F1-beräkningen
    denominator = precision + recall
    f1 = 2 * precision * recall / denominator if denominator > 0 else 0.0
    
    return precision, recall, f1

def calculate_overall_metrics(df_gold_entities, df_pred_entities, run_id):
    """
    Beräknar övergripande Precision, Recall och F1-score för alla entiteter.
    """
    # Merge för att hitta matchande entiteter (True Positives)
    # Matchning kräver identisk doc_id, label, start, end och text.
    df_matched = pd.merge(
        df_pred_entities.drop(columns=['source']), # Ta bort 'source' för merge
        df_gold_entities.drop(columns=['source']),
        how="inner", 
        on=["doc_id", "label", "start", "end", "text"]
    )
    
    true_positives = len(df_matched)
    total_predictions = len(df_pred_entities)
    total_gold = len(df_gold_entities)
    
    precision, recall, f1 = calculate_precision_recall_f1(
        true_positives, total_predictions, total_gold
    )
    
    # Grundläggande översiktsrad
    overview_row = {
        "run_id": run_id,
        "total_gold": total_gold,
        "total_predicted": total_predictions,
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    return overview_row

def calculate_per_label_f1(df_gold_entities, df_pred_entities, overview_row):
    """
    Beräknar F1-score separat för varje entitetstyp (label) och uppdaterar overview_row.
    """
    # Hitta alla unika labels från både guld- och prediktionsdata
    all_labels = set(df_gold_entities['label'].unique()).union(df_pred_entities['label'].unique())
    
    for label in sorted(all_labels):
        # Filtrera DataFrames för aktuell label
        df_gold_l = df_gold_entities[df_gold_entities['label'] == label]
        df_pred_l = df_pred_entities[df_pred_entities['label'] == label]
        
        # Hitta True Positives för denna label
        df_matched_l = pd.merge(
            df_pred_l.drop(columns=['source']),
            df_gold_l.drop(columns=['source']),
            how="inner", 
            on=["doc_id", "label", "start", "end", "text"]
        )
        
        tp_l = len(df_matched_l)
        total_predictions_l = len(df_pred_l)
        total_gold_l = len(df_gold_l)
        
        _, _, f1_l = calculate_precision_recall_f1(
            tp_l, total_predictions_l, total_gold_l
        )
        
        # Lägg till F1-score för denna label i översiktsraden
        overview_row[f"{label}_f1"] = f1_l

    return overview_row

# --- Huvudutvärderingslogik ---

def evaluate_ner_run(gold_data, prediction_data, run_id="baseline"):
    """
    Utför utvärderingen av Named Entity Recognition (NER) för ett dataset.
    Returnerar två DataFrames: en för övergripande och per-label metriker,
    och en för detaljerade per-dokument metriker.
    """
    all_gold_entities = []
    all_predicted_entities = []
    detailed_metrics_rows = []

    # 1. Iterera över dokument och normalisera entiteter
    for gold_doc, pred_doc in zip(gold_data, prediction_data):
        doc_id = gold_doc["id"]
        document_text = gold_doc["text"]
        
        gold_list = gold_doc.get("gold_entities", [])
        pred_list = pred_doc.get("predicted_entities", [])

        # Normalisera till en platt lista (för övergripande metrik)
        all_gold_entities.extend(normalize_entities(gold_list, doc_id, "gold"))
        all_predicted_entities.extend(normalize_entities(pred_list, doc_id, "prediction"))

        # Beräkna True Positives för den detaljerade dokumentrapporten
        # Använder set för att snabbt hitta unika matchande entiteter
        gold_set = {(e['label'], e['start'], e['end'], e['text']) for e in gold_list}
        pred_set = {(e['label'], e['start'], e['end'], e['text']) for e in pred_list}
        true_positives_doc = len(gold_set & pred_set)

        # Lägg till rad för den detaljerade rapporten
        detailed_metrics_rows.append({
            "run_id": run_id,
            "doc_id": doc_id,
            "text": document_text,
            "gold_count": len(gold_list),
            "predicted_count": len(pred_list),
            "true_positives": true_positives_doc,
            "notes": ""
        })

    # 2. Skapa DataFrames för alla entiteter
    df_gold = pd.DataFrame(all_gold_entities)
    df_pred = pd.DataFrame(all_predicted_entities)

    # 3. Beräkna övergripande metrik
    if df_gold.empty and df_pred.empty:
        # Hantera fallet där båda är tomma
        overview_row = {
            "run_id": run_id, 
            "total_gold": 0, "total_predicted": 0, "true_positives": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0
        }
    else:
        overview_row = calculate_overall_metrics(df_gold, df_pred, run_id)
        # 4. Beräkna per-label F1 och uppdatera översiktsraden
        overview_row = calculate_per_label_f1(df_gold, df_pred, overview_row)


    df_overview = pd.DataFrame([overview_row])
    df_detailed = pd.DataFrame(detailed_metrics_rows)

    return df_overview, df_detailed

# --- Kör scriptet ---

def main():
    """
    Huvudfunktion för att köra utvärderingsscriptet från kommandoraden.
    """
    # 1. Hantera kommandoradsargument
    if len(sys.argv) < 3:
        print("Användning: python eval_ner_2.py <gold_file.json> <predictions_file.json> [run_id]", file=sys.stderr)
        sys.exit(1)
        
    gold_file_path = sys.argv[1]
    predictions_file_path = sys.argv[2]
    run_id = sys.argv[3] if len(sys.argv) > 3 else "baseline"
    
    # Sätt standardfilnamn för utdata
    overview_output_file = f"overview_{run_id}.csv"
    detailed_output_file = f"detailed_{run_id}.csv"

    print(f"--- Startar NER-utvärdering för run: '{run_id}' ---")

    # 2. Läs in data
    gold_data = load_json_data(gold_file_path)
    prediction_data = load_json_data(predictions_file_path)

    # 3. Utför utvärderingen
    df_overview, df_detailed = evaluate_ner_run(
        gold_data, prediction_data, run_id=run_id
    )
    
    # 4. Spara resultaten
    save_dataframe_to_csv(df_overview, overview_output_file)
    save_dataframe_to_csv(df_detailed, detailed_output_file)

    print("--- Utvärdering klar ---")
    print(f"Övergripande metriker sparade till: {overview_output_file}")
    print(f"Detaljerade per-dokument metriker sparade till: {detailed_output_file}")

if __name__ == "__main__":
    main()