import json

in_file = "data-sv-200.json"
out_file = "data-sv-200.txt"
	
def convert_doc_to_toon(doc):
	doc_id = doc["id"]
	language = doc["language"]
	text = doc["text"]
	
	# Document header - always the same
	header_line = "doc{id,language,text}:"
	
	# Data line with variables
	data_line = f'  "{doc_id}","{language}","{text}"'
	
	# Store lines in list
	lines = []
	lines.append(header_line)
	lines.append(data_line)
	
	k = len(doc["gold_entities"]) # Number of gold entities in the object
	entity_header_line = f'\n doc.gold_entities[{k}]{{label,start,end,text}}:' # Build the entity header line
	
	lines.append(entity_header_line)
	
	for entity in doc["gold_entities"]: # Loop over gold entities in json to extract values
		label = entity["label"]
		start = entity["start"]
		end = entity["end"]
		text = entity["text"]
		
		row = f'  "{label}",{start},{end},"{text}"'
		
		lines.append(row)

	
	# Combine lines into a single string
	toon_text = "\n".join(lines)
	
	return toon_text


with open(in_file, "r") as f:
	data = json.load(f) # Full json file loaded into data

# Store the converted TOON objects
toon_list = []

# Loop over each JSON doc to convert it to TOON using the function above
for doc in data: 
	toon_doc = convert_doc_to_toon(doc)
	toon_list.append(toon_doc)
	
 # Combine into one string
toon_docs = "\n\n".join(toon_list)

# Write to file
with open(out_file, "w", encoding="utf-8") as f:
	f.write(toon_docs)