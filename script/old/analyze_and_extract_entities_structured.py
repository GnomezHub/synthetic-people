import json

def analyze_and_extract_entities_structured(text, target_entity_list):
    """
    Analyzes a text to find occurrences of specific entities defined in a structured list.
    
    The target_entity_list elements start with a code (1-4) followed by the text.
    It uses a greedy approach (longest match first) and ensures no overlaps.
    """
    
    # 1. Define Constants and Mappings (LABEL_MAP in English as per request)
    LABEL_MAP = {
        '1': 'NAME',
        '2': 'PHONE',
        '3': 'ADDRESS',
        '4': 'NATIONAL_ID'
    }

    # 2. Parse and Prepare Input List
    parsed_requests = []
    # The actual text to search for, stored with its label code
    requested_texts_and_codes = [] 
    
    for i, item in enumerate(target_entity_list):
        if len(item) < 2 or item[0] not in LABEL_MAP:
            print(f"Warning: Skipping invalid entity input '{item}' in list.")
            continue
            
        code = item[0]
        entity_text = item[1:]
        
        # Store for chronological matching
        parsed_requests.append({
            'code': code,
            'text': entity_text,
            'is_matched': False
        })
        # Store for finding unique texts (used in greedy search)
        requested_texts_and_codes.append((entity_text, code))

    # Get unique texts, sorted by length (longest first) for the greedy search
    unique_texts = sorted(list(set(item['text'] for item in parsed_requests)), key=len, reverse=True)
    
    # 3. Find All Non-Overlapping Occurrences (Greedy Search)
    found_occurrences = []
    occupied_indices = set()

    for entity in unique_texts:
        search_start = 0
        while True:
            found_start = text.find(entity, search_start)
            
            if found_start == -1:
                break
            
            found_end = found_start + len(entity)
            
            # Check for collision
            is_occupied = any(i in occupied_indices for i in range(found_start, found_end))
            
            if not is_occupied:
                # Valid match found
                found_occurrences.append({
                    "text": entity,
                    "start": found_start,
                    "end": found_end,
                    "matched_request_index": -1 # Will be set during matching
                })
                
                # Mark indices as occupied
                for i in range(found_start, found_end):
                    occupied_indices.add(i)
            
            # Move the search cursor forward
            search_start = found_start + 1

    # 4. Match Found Occurrences to Requested Entities & Generate Output
    final_results = []
    entity_id_counter = 1
    
    # Sort found occurrences chronologically for correct ID assignment
    found_occurrences.sort(key=lambda x: x['start'])

    for found_item in found_occurrences:
        matched_request = None
        
        # Try to find an unmatched request that corresponds to this found text
        for idx, req in enumerate(parsed_requests):
            if not req['is_matched'] and req['text'] == found_item['text']:
                matched_request = req
                req['is_matched'] = True
                break
        
        # Use the matched request's code and label, or assign "EXTRA" if no request was pending
        if matched_request:
            code = matched_request['code']
            label = LABEL_MAP[code]
        else:
            # Handle "EXTRA" occurrence
            # Since the text and code are unknown, we must use a placeholder/default
            code = '?'
            label = f"EXTRA ({found_item['text']})" # Use text as placeholder label

        # Create the final result object
        final_results.append({
            "id": f"e{entity_id_counter}",
            "label": label,
            "start": found_item['start'],
            "end": found_item['end'],
            "text": found_item['text']
        })
        entity_id_counter += 1

    # 5. Reporting
    print("--- Analysis Report ---")
    status_ok = True
    
    # Check for MISSING requests
    for req in parsed_requests:
        if not req['is_matched']:
            label_str = LABEL_MAP[req['code']]
            print(f"[MISSING] Expected '{req['text']}' with label {label_str}, but no available position was found in text.")
            status_ok = False
            
    # Check for EXTRA occurrences (those marked with 'EXTRA' label above)
    extra_count = sum(1 for item in final_results if item['label'].startswith('EXTRA'))
    if extra_count > 0:
        print(f"[EXTRA] Found {extra_count} occurrences in text that were not explicitly requested in the list (added to JSON).")
        status_ok = False

    if status_ok:
        print("Success: All requested entities were found and matched.")
    print("-----------------------")

    # Return the final JSON string
    return json.dumps(final_results, indent=2, ensure_ascii=False)





source_text = "IT-support fick ett ärende där användaren skickat in sitt telefonnummer 0722 55 44 90 och sin adress: Lillgatan 7, 702 12 Örebro."
entities = ["20722 55 44 90", "3Lillgatan 7, 702 12 Örebro"] 
json_output = analyze_and_extract_entities_structured(source_text, entities)
print(f"source_text:{source_text}")
print("\nJSON Output:")
print(json_output)