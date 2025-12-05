import streamlit as st
import ollama
import json


# pip install streamlit

MODEL_NAME = "gemma3:4b"
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

# --- DINA KÄRNFUNKTIONER ---

def prompt_model(text):
	"""Anropar LLM och returnerar prediktioner och eventuellt fel."""
	user_prompt = f"Extract all entities from the following text and respond only with the requested entity string:\n\n{text}"
	try:
		response = ollama.chat(
			model = MODEL_NAME,
			messages=[
				{"role": "system", "content": SYSTEM_PROMPT},
				{"role": "user", "content": user_prompt}
			],
			options={"temperature": 0.1}
		)
		raw_response = response['message']['content']
	except Exception as e:
		return None, str(e)

	lines = raw_response.splitlines()
	seen = set()
	entities = []

	for line in lines:
		line = line.strip()
		if len(line) < 2 or not line[0].isdigit(): continue
		label_id = line[0]
		if label_id not in LABEL_IDS or line in seen: continue
		
		seen.add(line)
		entities.append((label_id, line[1:])) 

	return entities, None

def index_finder(text, entity_texts):
	"""Din index_finder-funktion (oförändrad logik)."""
	found_entities = []
	occupied_indices = set()
	unique_entities = sorted(list(set(entity_texts)), key=len, reverse=True)

	for entity_text in unique_entities:
		search_start = 0
		while True:
			start_index = text.find(entity_text, search_start)
			if start_index == -1: break
			
			end_index = start_index + len(entity_text)
			is_occupied = any(i in occupied_indices for i in range(start_index, end_index))

			if is_occupied:
				search_start += 1
				continue
			else:
				found_entities.append({"text": entity_text, "start": start_index, "end": end_index})
				for i in range(start_index, end_index): occupied_indices.add(i)
				search_start = end_index
	
	found_entities.sort(key=lambda x: x['start'])
	return found_entities

def build_json(predictions, indexed_entities):
	"""Din build_json-funktion (oförändrad logik)."""
	label_map = {'1': 'NAME', '2': 'PHONE', '3': 'ADDRESS', '4': 'NATIONAL_ID', '5': 'EMAIL'}
	final_entities = []
	index_lookup = {item["text"]: item for item in indexed_entities}
 
	for label_id, entity_text in predictions:
		if entity_text in index_lookup:
			index_info = index_lookup[entity_text]
			label_str = label_map.get(label_id, "Unknown")
			
			final_entities.append({
				"label": label_str,
				"start": index_info["start"],
				"end": index_info["end"],
				"text": entity_text
			})
	return final_entities

# --- STREAMLIT HUVUDFLOW ---

st.set_page_config(page_title="PII Extractor (Streamlit)", layout="wide")
st.title("🤖 PII Extractor med Streamlit och Ollama")
st.caption(f"Använder modell: `{MODEL_NAME}`")

uploaded_file = st.file_uploader("Välj din JSON-datafil:", type=['json'])

if uploaded_file is not None:
	# Läs in filen
	try:
		# Streamlit läser filen som en BytesIO, vi behöver dekoda den
		string_data = uploaded_file.getvalue().decode("utf-8")
		data = json.loads(string_data)
		st.success(f"Filen '{uploaded_file.name}' laddades upp. Innehåller {len(data)} dokument.")
	except json.JSONDecodeError:
		st.error("Kunde inte avkoda JSON. Kontrollera filformatet.")
		data = None

	if data is not None and st.button("Starta Extraktion och Maskering"):
		
		# Skapar en placeholder för loggar som uppdateras under körningen
		log_placeholder = st.empty()
		
		output_docs = []
		total_docs = len(data)

		# Skapa en Streamlit-progress bar
		progress_bar = st.progress(0)
		
		with log_placeholder.container():
			st.subheader("Processlogg")
			# Denna lista kommer att agera som vår logg
			log_lines = []
			
			# Streamlit har en speciell funktion för att iterera över långa listor
			# och uppdatera UI utan att fastna, men för enkelhetens skull kör vi en vanlig loop
			# och uppdaterar loggen inuti loopen.
			
			for i, doc in enumerate(data):
				doc_id = doc.get("id", "Unknown")
				text = doc.get("text", "")

				# Uppdatera loggen
				log_lines.append(f"[{i+1}/{total_docs}] Bearbetar dokument ID: {doc_id}...")
				
				# Anropa modell
				predictions, error = prompt_model(text)

				if error:
					log_lines.append(f" -> ERROR vid anrop till modell: {error}")
					# För att visa loggen direkt i gränssnittet, använd st.code eller st.text
					st.code("\n".join(log_lines), language='text')
					continue

				if predictions is None:
					log_lines.append(" -> Inga prediktioner eller fel vid parsing.")
					st.code("\n".join(log_lines), language='text')
					continue
				
				# --- Bearbetning ---
				entity_texts = [et for (_, et) in predictions]
				indexed = index_finder(text, entity_texts)
				predicted_entities = build_json(predictions, indexed)

				count = len(predicted_entities)
				log_lines.append(f" -> Klart. Hittade {count} entiteter.")

				# Konfigurera output
				output_doc = {
					"id": doc_id, 
					"language": doc.get("language", ""),
					"text": text, 
					"predicted entities": predicted_entities
				}
				output_docs.append(output_doc)
				
				# Uppdatera progress bar
				progress_bar.progress((i + 1) / total_docs)
				
				# Uppdatera loggen i UI (Viktigt steg för att se realtid)
				st.code("\n".join(log_lines), language='text')

		st.success(f"Extraktion klar! {len(output_docs)} dokument bearbetade.")
		
		# 3. Visa och tillhandahålla nedladdning
		st.subheader("Resultat")
		
		# Nedladdningsknapp
		json_output = json.dumps(output_docs, ensure_ascii=False, indent=2)
		st.download_button(
			label="Ladda ner predictions som JSON",
			data=json_output,
			file_name="predictions_streamlit.json",
			mime="application/json",
		)
		
		# Visa de första 5 dokumenten för granskning
		st.markdown("##### Första 5 resultaten (förhandsgranskning):")
		st.json(output_docs[:5])