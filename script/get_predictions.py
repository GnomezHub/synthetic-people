import ollama
import json
import sys
import os

MODEL_NAME = "gemma3:4b"

# Later change this to define files in terminal
INPUT_FILE = "data-test.json"
OUTPUT_FILE = "predictions-test.json"

LABEL_IDS = {"1", "2", "3", "4", "5"}

SYSTEM_PROMPT = """
You are extracting specific entities from text. 
You must identify ONLY the following labels, and return ONLY their numeric IDs:

1 = NAME  
2 = PHONE  
3 = ADDRESS  
4 = NATIONAL_ID
5 = EMAIL

INSTRUCTIONS:
- Read the user text.
- Find every entity that matches one of the labels above. Do not include any labels that are not present in the list above.
- For each entity, output EXACTLY ONE LINE, formatted as:

    <label_id><entity_text>

Example text: "När handläggaren på Skatteverket ringde stod det att ansökan skickats av Elin Rask. Hennes nummer är 0722 33 44 55."
The correct output for this text is:

    1Elin Rask
    20722 33 44 55

- The EMAIL label must be used for email addresses. They always contain an "@". You have to mark them as EMAIL (5), noting else. Even if they contain one or more names, they have to be marked as EMAIL.

Example: "En lärare rapporterade att e-postadressen sara.lindgren@edu.se inte gick att nå."
The correct output for this text is:

	5sara.lindgren@edu.se

- Do NOT include commas, colons, JSON, quotes or explanations. Only the label id and entity text.
- If no entities are found, output an empty string.

Your entire response must consist ONLY of these lines, nothing else.

"""

def load_data(filename):
	"""Read data from a JSON file"""
	if not os.path.exists(filename):
		print(f'ERROR: File "{filename}" could not be found.')
		sys.exit(1) # Interrupt the script
	
	try:
		with open(filename, "r", encoding="utf-8") as f:
			return json.load(f)
	
	except json.JSONDecodeError:
		print(f'ERROR: JSON could not be decoded from "{filename}"')
		sys.exit(1)
		
	except Exception as e:
		# Catch any other exception
		print(f'ERROR: An unexpected error occurred while reading "{filename}": {e}')
		sys.exit(1)


def save_data(filename, data):
	"""Save data to a JSON file"""
	try:
		with open(filename, "w", encoding="utf-8") as f:
			json.dump(data, f, ensure_ascii=False, indent=2)
	
	except IOError as e:
		print(f'ERROR: Could not write to file "{filename}". Error: {e}')
	
	except Exception as e:
		print(f'ERROR: An unexpected error occurred: {e}')	

def prompt_model(text):
	"""Prompts the LLM for one document and returns a list of tuples with label id and entity."""

	user_prompt = f"Extract all entities from the following text and respond only with the requested entity string:\n\n{text}"

	try:
		response = ollama.chat(
			model = MODEL_NAME,
			messages=[
				{"role": "system", "content": SYSTEM_PROMPT},
				{"role": "user", "content": user_prompt}
			],
			options={
				"temperature": 0.1
			}
		)

		raw_response = response['message']['content']

	except Exception as e:
		print(f"ERROR: Model call failed: {e}")
		return None

	lines = raw_response.splitlines()

	seen = set() # Ensure unique lines
	entities = []

	for line in lines:
		line = line.strip() # Strip any surrounding whitespace
		if len(line) < 2:
			print(f"WARNING: Very short or empty line detected. Skipping line: {line}")
			continue
		if not line[0].isdigit():
			print(f"WARNING: Line does not start with index. Skipping line: {line}")
			continue
		label_id = line[0]
		if label_id not in LABEL_IDS:
			print(f"Unknown label index detected. Skipping line: {line}")
			continue
		if line in seen:
			print(f"WARNING: Duplicate entity detected. Skipping line: {line}")
			continue

		seen.add(line)
		entities.append((label_id, line[1:])) # Appends a tuple for each line: [(1, Elin Rask), (2, 0722 33 44 55)]

	return entities

def index_finder(text, entity_texts):
	"""
	Takes the text and a list of entity substrings as input. 
	For each entity, finds the start and end index in the text. 
	Ensures no overlaps if there are identical entities ("Mia gillar att heta Mia.").
	Returns a list of dictionaries: {"text": ..., "start"..., "end"...,}
	"""
	found_entities = []
	occupied_indices = set()

	# Sort the list by length (descending) to prioritize longest entities first
	unique_entities = sorted(list(set(entity_texts)), key=len, reverse=True)

	for entity_text in unique_entities:
		search_start = 0
		while True:
			start_index = text.find(entity_text, search_start) # Finds first occurrence of entity_text after search_start

			if start_index == -1:
				break # No more occurrences found

			else:
				end_index = start_index + len(entity_text)

				is_occupied = any(i in occupied_indices for i in range(start_index, end_index))

				if is_occupied: # Index already occupied (found), move search cursor forward one step
					search_start += 1
					continue
				else:
					found_entities.append({
						"text": entity_text,
						"start": start_index,
						"end": end_index
					})
					
				# Mark these positions as occupied
				for i in range(start_index, end_index):
					occupied_indices.add(i)

				# Move search cursor forward
				search_start = end_index
	
	# Sort the list by start_index for a consistent order
	found_entities.sort(key=lambda x: x['start'])

	return found_entities

def build_json(predictions, indexed_entities):
	"""
    Takes:
        predictions = [(label_id, entity_text), ...]
        indexed_entities = [ {"text": ..., "start": ..., "end": ...}, ...]

    Returns:
        A list of JSON dictionaries matching the gold format:
		[{'label': 'NAME', 'start': 23, 'end': 27, 'text': 'Elin Rask'}, {'label': 'PHONE', 'start': 12, 'end': 48, 'text': '0722 33 44 55'}]
    """

	label_map = {
        '1': 'NAME',
        '2': 'PHONE',
        '3': 'ADDRESS',
        '4': 'NATIONAL_ID',
		'5': 'EMAIL'
    }

	final_entities = []
 
	index_lookup = {}
	for item in indexed_entities:
		text = item["text"]
		index_lookup.setdefault(text, []).append(item)

 
	for label_id, entity_text in predictions:
		index_list = index_lookup.get(entity_text)

		if not index_list or len(index_list) == 0:
			print(f"WARNING: No remaining index match for entity '{entity_text}'")
			continue

		index_info = index_list.pop(0)

		# Convert label_id -> label string
		label_str = label_map.get(label_id, "Unknown")

		entity_obj = {
			"label": label_str,
			"start": index_info["start"],
			"end": index_info["end"],
			"text": entity_text
		}

		final_entities.append(entity_obj)
	
	return final_entities

# A main loop that processes all documents and saves the results to a file
def main():
	docs = load_data(INPUT_FILE)

	output_docs = []

	for doc in docs:
		text = doc.get("text", "")
		doc_id = doc.get("id", "Unknown") # E.g. "sv-001"

		print(f"Processing document: {doc_id}")

		# Call the model
		predictions = prompt_model(text)
		if predictions is None:
			print(f"ERROR: Model failed to process document {doc_id}. Skipping.")
			continue

		# Extract only the text part for index finder
		entity_texts = [entity_text for (_, entity_text) in predictions]

		# Pass text values to index finder
		indexed = index_finder(text, entity_texts)

		# Build JSON entity objects
		predicted_entities = build_json(predictions, indexed)

		# Construct final document object
		output_doc = {
			"id": doc_id, 
			"language": doc.get("language", ""),
			"text": text, 
			"predicted entities": predicted_entities
		}

		output_docs.append(output_doc)

		# Save everything to file
		save_data(OUTPUT_FILE, output_docs)

	print(f"Done! Predictions saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
	main()