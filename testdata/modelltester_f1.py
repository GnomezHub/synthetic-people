#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 11:21:16 2025

@author: marieonette
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 12:41:48 2025

@author: Danny Gomez

UPPDATERAD: F1-validering nu integrerad. Använder entiteterna i INPUT_FILE 
som "gold standard" för jämförelse.
"""

import ollama
import json
import sys
import os

# --- Konfiguration ---
MODEL_NAME = "gemma3:4b"  # Se till att du har denna modell: `ollama pull gemma3:4b`
INPUT_FILE = "testdata_fullname.json" # Filen som innehåller BÅDE råtext OCH de sanna 'gold_entities'
OUTPUT_FILE = "testdata_fullname_predicted.json" # Fil för att spara modellens prediktioner

# Globala variabler för ackumulering av F1-statistik
GLOBAL_TP = 0
GLOBAL_FP = 0
GLOBAL_FN = 0

# --- System-prompt för Modellen ---
SYSTEM_PROMPT = """
Du är en expert på att extrahera information (Named Entity Recognition - NER) från svensk text.
Din uppgift är att hitta och extrahera specifika entiteter från texten som användaren tillhandahåller.

Du MÅSTE använda **exakt** följande etiketter:
- FIRST_NAME: Ett separat förnamn.
- LAST_NAME: Ett separat efternamn.
- FULL_NAME: Ett komplett namn (för- och efternamn) när de förekommer tillsammans.
- ADDRESS: En gatuadress, postnummer och stad.
- PHONE: Ett telefonnummer.
- EMAIL: En e-postadress.
- NATIONAL_ID: Ett svenskt personnummer (t.ex. ÅÅMMDD-XXXX).

VIKTIG REGEL FÖR NAMN:
Använd `FULL_NAME` om du hittar ett för- och efternamn och eventuella mellannamn direkt efter varandra (t.ex. 'Anna Karlsson' eller 'Monika Marie Grönlund') .
Använd `FIRST_NAME` och `LAST_NAME` endast om de förekommer helt separat (t.ex. 'Patientens namn: Erik' eller 'Dr. Svensson').

OUTPUT-FORMAT:
Du måste svara med **enbart** en JSON-lista. Inkludera ingen förklarande text, inga ursäkter, inga kommentarer - bara JSON.
Varje objekt i listan ska ha följande exakta struktur:
{
  "label": "ETIKETT_NAMN",
  "text": "den extraherade texten",
  "start": start_index_i_texten,
  "end": end_index_i_texten
}

Om du inte hittar några entiteter alls, svara med en tom lista: []
"""
# ------------------------------------

def load_data(filename):
    """Läser in JSON-datan från en fil."""
    if not os.path.exists(filename):
        print(f"FEL: Inputfilen '{filename}' hittades inte.", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"FEL: Kunde inte avkoda JSON från '{filename}'. Filen kan vara korrupt.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"FEL vid läsning av fil: {e}", file=sys.stderr)
        sys.exit(1)

def save_data(filename, data):
    """Sparar datan till en JSON-fil."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nResultat sparat till '{filename}'")
    except IOError as e:
        print(f"FEL: Kunde inte skriva till filen '{filename}'. Fel: {e}", file=sys.stderr)

def get_entities_from_model(text_content):
    """Anropar Ollama-modellen för att extrahera entiteter."""
    
    user_prompt = f"Extrahera alla entiteter från följande text och svara **endast** med den begärda JSON-listan:\n\n{text_content}"
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.0  # Låg temperatur för mer konsekventa JSON-svar
            }
        )
        
        raw_response = response['message']['content'].strip()
        
        # Rensa bort eventuella "markdown" ```json ... ```
        if raw_response.startswith("```json"):
            start_index = raw_response.find('\n') 
            if start_index == -1 or start_index > 10:
                start_index = 7
            raw_response = raw_response[start_index:].strip()
            
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3].strip()
        
        # Försök att parsa JSON
        entities = json.loads(raw_response)
        return entities

    except json.JSONDecodeError:
        print("\n--- VARNING: Kunde inte parsa JSON-svar från modellen ---", file=sys.stderr)
        print(f"Modellens råa svar (delvis): {raw_response[:200]}...", file=sys.stderr)
        return [] # Returnera en tom lista vid fel
    except Exception as e:
        # Fångar anslutningsfel till Ollama
        print(f"\n--- FEL: Något gick fel vid anrop till Ollama ({e}) ---", file=sys.stderr)
        print(f"Kontrollera att Ollama-servern körs och att modellen '{MODEL_NAME}' är nedladdad.", file=sys.stderr)
        return None # Returnera None för att signalera ett allvarligt fel

def calculate_f1_score(gold_entities_raw, model_predictions_raw):
    """
    Beräknar precision, recall och F1-score baserat på exakt matchning.
    
    gold_entities_raw: Lista med tuples (label, start, end) från guldstandarden.
    model_predictions_raw: Lista med tuples (label, start, end) från modellens prediktioner.
    
    Returnerar (TP, FP, FN).
    """
    
    # Konvertera till sets för effektiv jämförelse.
    gold_set = set(gold_entities_raw)
    pred_set = set(model_predictions_raw)
    
    # True Positives (TP): Entiteter som finns i både guld och prediktion.
    TP = len(gold_set.intersection(pred_set))
    
    # False Positives (FP): Entiteter som modellen hittade, men som inte finns i guld.
    FP = len(pred_set - gold_set)
    
    # False Negatives (FN): Entiteter i guld som modellen missade.
    FN = len(gold_set - pred_set)
    
    return TP, FP, FN

def process_data(input_data):
    """Itererar genom indatan, anropar modellen, bygger upp utdatan och beräknar F1-statistik."""
    global GLOBAL_TP, GLOBAL_FP, GLOBAL_FN
    output_data = []
    
    for i, item in enumerate(input_data):
        item_id = item.get('id', i+1)
        print(f"Bearbetar nod {item_id}... ", end="")
        text_to_analyze = item['text']
        
        # 1. Förbered Guld-entiteterna för F1-jämförelse
        gold_f1_list = []
        for entity in item.get('gold_entities', []):
             # Vi använder en tuple som nyckel för att entiteten ska vara hashbar och jämförbar
             # Säkerställ att start och end är integers
            try:
                gold_f1_list.append(
                    (entity['label'], int(entity['start']), int(entity['end']))
                )
            except (ValueError, KeyError):
                print(f"\n--- VARNING: Ogiltigt format i guld-entitet för nod {item_id}.", file=sys.stderr)
                continue

        # 2. Få entiteter från modellen
        model_entities = get_entities_from_model(text_to_analyze)
        
        if model_entities is None:
            print("Avbryter på grund av tidigare fel.")
            return None # Avbryt hela processen

        # 3. Formatera om modellens resultat och förbered för F1-jämförelse
        formatted_entities = []
        model_f1_list = []
        
        for j, entity in enumerate(model_entities):
            if not all(k in entity for k in ('label', 'text', 'start', 'end')):
               print(f"\n--- VARNING: Hoppar över felaktigt formaterad entitet från modell: {entity}", file=sys.stderr)
               continue

            start, end, text, label = entity['start'], entity['end'], entity['text'], entity['label']
            
            # Validera och konvertera index
            try:
                start, end = int(start), int(end)
                if start < 0 or end > len(text_to_analyze) or start >= end:
                    raise ValueError("Index ogiltiga eller utanför gränsen.")
            except ValueError as ve:
                print(f"\n--- VARNING: Ogiltiga index/typ: {ve} i nod {item_id}.", file=sys.stderr)
                continue
                 
            # Indexmatchning
            predicted_text_slice = text_to_analyze[start:end]
            if predicted_text_slice != text:
                print(f"\n--- VARNING: Modellens index matchar inte text! '{text}' != '{predicted_text_slice}'.", file=sys.stderr)
                # Fallback: försök hitta texten manuellt (kan vara mindre tillförlitligt)
                new_start = text_to_analyze.find(text)
                if new_start != -1:
                    start = new_start
                    end = new_start + len(text)
                else:
                    print(f"--- VARNING: Kunde inte hitta '{text}'. Hoppar över entitet.", file=sys.stderr)
                    continue

            # Lägg till i listan för den sparade output-filen
            formatted_entities.append({
                "id": f"e{j+1}", 
                "label": label,
                "start": start,
                "end": end,
                "text": text
            })
            
            # Lägg till i listan för F1-jämförelse
            model_f1_list.append((label, start, end))

        # 4. Beräkna F1-statistik för denna nod
        TP, FP, FN = calculate_f1_score(gold_f1_list, model_f1_list)
        
        # 5. Ackumulera till globala variabler
        GLOBAL_TP += TP
        GLOBAL_FP += FP
        GLOBAL_FN += FN
        
        # 6. Bygg det nya objektet för output-filen
        # Obs: Vi sparar modellens prediktioner som 'gold_entities' i outputfilen
        # för att följa formatet från den ursprungliga koden.
        new_item = {
            "id": item_id,
            "language": item['language'],
            "text": item['text'],
            "gold_entities": formatted_entities 
        }
        output_data.append(new_item)
        print(f"Klar. Hittade {len(formatted_entities)} entiteter. (TP: {TP}, FP: {FP}, FN: {FN})")
        
    return output_data

def calculate_overall_f1():
    """Beräknar och skriver ut den totala F1-statistiken."""
    global GLOBAL_TP, GLOBAL_FP, GLOBAL_FN

    print("\n" + "="*50)
    print("  SAMMANFATTNING AV NAMED ENTITY RECOGNITION")
    print("="*50)
    print(f"Totalt antal Guld-entiteter (TP + FN): {GLOBAL_TP + GLOBAL_FN}")
    print(f"Totalt antal Predikterade entiteter (TP + FP): {GLOBAL_TP + GLOBAL_FP}")
    print("-"*50)
    print(f"  True Positives (TP): {GLOBAL_TP}")
    print(f"  False Positives (FP): {GLOBAL_FP}")
    print(f"  False Negatives (FN): {GLOBAL_FN}")
    print("="*50)
    
    # Förhindra division med noll
    precision = GLOBAL_TP / (GLOBAL_TP + GLOBAL_FP) if (GLOBAL_TP + GLOBAL_FP) > 0 else 0.0
    recall = GLOBAL_TP / (GLOBAL_TP + GLOBAL_FN) if (GLOBAL_TP + GLOBAL_FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1_score:.4f}")
    print("="*50)

# --- Huvudprogram ---
def main():
    print(f"Startar bearbetning med modell: {MODEL_NAME}")
    print(f"Läser indata (med guldstandard) från: {INPUT_FILE}")
    
    # 1. Läs in indata (som innehåller både text och guldstandard)
    input_data = load_data(INPUT_FILE)
    if not input_data:
        return

    # 2. Bearbeta datan och få prediktioner (jämförelse mot guldstandarden sker här)
    predictions = process_data(input_data)
    
    # 3. Spara utdata
    if predictions:
        save_data(OUTPUT_FILE, predictions)
    else:
        print("Ingen utdata genererades på grund av fel.", file=sys.stderr)
        
    # 4. Beräkna och skriv ut total F1-score
    calculate_overall_f1()

if __name__ == "__main__":
    main()