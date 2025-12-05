def build_json(predictions, indexed_entities):
	"""
    Takes:
        predictions = [(label_id, entity_text), ...]  # Kan innehålla dubbletter
        indexed_entities = [ {"text": ..., "start": ..., "end": ...}, ...] # Alla unika index för alla texter

    Returns:
        A list of JSON dictionaries matching the gold format:
		[{'label': 'NAME', 'start': 23, 'end': 27, 'text': 'Elin Rask'}, ...]
    """

	label_map = {
        '1': 'NAME',
        '2': 'PHONE',
        '3': 'ADDRESS',
        '4': 'NATIONAL_ID',
		'5': 'EMAIL'
    }

	# Skapa en 'index-bank' där varje unik entitetstext mappar till en lista av dess index-objekt.
	# Exempel: {"Elin Rask": [obj1, obj2], "0722 33 44 55": [obj3]}
	index_bank = {}
	for item in indexed_entities:
		text_key = item["text"]
		if text_key not in index_bank:
			index_bank[text_key] = []
		# Lägg till hela index-objektet i listan.
		index_bank[text_key].append(item)

	final_entities = []

	# Loopa över LLM:s prediktioner. Dessa kan innehålla dubbletter.
	for label_id, entity_text in predictions:
		
		# Hämta listan med index-objekt för den aktuella entitetstexten
		index_list = index_bank.get(entity_text)
		
		if not index_list:
			# Detta ska inte hända om index_finder fungerar korrekt, men är en säkerhetsåtgärd.
			print(f"WARNING: Hittade ingen index-information för entiteten: {entity_text}")
			continue

		# Använd DET FÖRSTA index-objektet i listan (FIFO).
		# Vi antar att ordningen i 'predictions' och 'indexed_entities' matchar.
		# Om 'indexed_entities' är sorterad efter 'start'-index (vilket den är i index_finder),
		# kommer den första i listan att vara den första förekomsten i texten.
		index_info = index_list.pop(0) 

		# Konvertera label_id -> label string
		label_str = label_map.get(label_id, "Unknown")

		entity_obj = {
			"label": label_str,
			"start": index_info["start"],
			"end": index_info["end"],
			"text": entity_text # Använd den exakta texten från LLM:s output
		}

		final_entities.append(entity_obj)
	
	return final_entities