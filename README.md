
## Sammanfattning:

Vi har utvecklat ett verktyg som låter användaren ladda upp en pdf i ett gränssnitt och får tillbaka en version av filen där känsliga personuppgifter maskerats. Identifieringen av personuppgifter är ett exempel på NER (Named Entity Recognition) med stora språkmodeller.

## Användning:
1. Klona github-repot:
```
git clone https://github.com/GnomezHub/synthetic-people.git
```

2. Installera de externa biblioteken Poppler (omvandlar pdf till bild) och Tesseract (omvandlar bild till text):

**MacOS (via Homebrew):**

```
brew install tesseract poppler
```
    
**Ubuntu/Linux:**

```
sudo apt update
sudo apt install tesseract-ocr poppler-utils
```

**Windows:**

Ladda ner binärer för Tesseract och Poppler och lägg till dem i din PATH.

3. Installera Python-bibliotek:
```
pip install flask openai pdfplumber pdf2image pytesseract pillow fpdf werkzeug
```

4. Ställ in API-nyckel:
**MacOS/Linux:**
```
export OPENAI_API_KEY='[nyckel]'
```

**Windows (PowerShell):**
```
$env:OPENAI_API_KEY='[nyckel]'
```

5. Navigera till rätt mapp:
```
cd flask
```

6. Starta applikationen:
```
python app.py
```

## Syfte

Syftet med projektet är att undersöka hur genomförbart det är att ta fram ett verktyg som, med hjälp av en stor språkmodell, kan identifiera känsliga personuppgifter i dokument och maskera dem korrekt.

## Repots innehåll
### Data/
Denna mapp innehåller den data vi tagit fram och använt för att utvärdera modeller under den första fasen. Gold-sv-30.json är en kortare version av gold-sv-200.json.

### Script/
Scripten i denna mapp är de som använts för modellutvärdering. 

get_predictions.py är det script som promptar modellen, ger den input, hittar start- och slutindex för identifierade entiteter och bygger upp JSON-objekt.
get_predictions_openai.py är samma, bara konfigurerat för OpenAI's API.

eval.py är det script som jämför båda JSON-filerna (vår gold-data och modellens predictions) och räknar ut precision, recall och f1. Den utvärderar på två olika sätt, vilket är väl förklarat i kommentarerna.

### Flask/
app.py och templates/index.html används för gränssnittet. 

## Arbetets gång
### 1. Data
Vi tog fram 200 meningar med syntetiska personuppgifter för att testa och utvärdera LLMs på uppgiften. Vi genererade meningar innehållande flera olika sorters personuppgifter:

- Namn
- Adress
- Telefonnummer
- Personnummer
- Email

Datan är syntetisk, alltså påhittad, men är utformad för att efterlikna verkliga exempel. Vi använde generativ AI för att hjälpa oss generera exempel.

Av de 200 meningarna byggde vi JSON-objekt i detta format:
```
{
    "id": "sv-001",
    "language": "sv",
    "text": "När handläggaren på Skatteverket ringde stod det att ansökan skickats av Elin Rask. Hennes nummer är 0722 33 44 55.",
    "gold_entities": [
      {
        "id": "e1",
        "label": "NAME",
        "start": 73,
        "end": 83,
        "text": "Elin Rask"
      },
      {
        "id": "e2",
        "label": "PHONE",
        "start": 101,
        "end": 114,
        "text": "0722 33 44 55"
      }
```

Vi använde kod för att korrekt hitta start- och slutindex för entiteterna, då vi upptäckte att LLM inte lyckades med det.

## 2. Modellutvärdering

Vi började med att testa modeller lokalt via Ollama, sedan gick vi över till att använda OpenAI.

Modellen  fick text-fältet (meningen) som input och fick instruktioner om att hitta entiteter i den, utifrån en fördefinierad lista med etiketter (samma som ovan). Den ombads ge sitt svar i en sträng med etikett och entitet, till exempel:
`1Elin Rask`
där den första siffran motsvarar en etikett. Vi kom fram till detta format, istället för att be modellen svara med ett helt JSON-objekt, då vi försökte minimera antalet tokens som skulle skickas över API.

Efter modellens svar använde vi kod för att hitta start- och slutindex för de entiteter som modellen identifierat, och bygga upp JSON-objektet utifrån det. Resultatet blir en JSON-fil som har exakt samma struktur som vår gold-fil. 

Med de två filerna (gold och predictions) kunde vi utvärdera hur väl modellen presterat genom att mäta dess precision och recall och väga samman det till ett f1-score.
Vi experimenterade med olika modeler och att ändra systemprompten för att se hur resultatet påverkades. Mätningarna dokumenterades i ett kalkylark, där den raden som är i fetstil markerar det bästa resultatet:

https://docs.google.com/spreadsheets/d/1SRryb4xJOOVl2xTvwc15Cf5zywOSJuQHn5MEB7T5TjA/edit?gid=1495298767#gid=1495298767

## 3. Gränssnitt

Gränssnittet är byggt med Flask och med html, javascript och bootstrap.

Användaren laddar upp en pdf-fil. Python-biblioteken används för att extrahera text från pdf:en. Texten delas upp i "chunks" för att underlätta för modellen, som får en chunk i varje prompt. Modellen svarar med entiteter den hittat, och deras etiketter. Gränssnittet visar vilka entiteter som hittats och vilka etiketter de tilldelats, genom färgkodning. Man kan sedan ladda ner filen där entiteterna är maskerade, t.ex: "Jag heter [namn] och bor på [adress]".

## Förslag på utveckling
En bra utveckling för framtiden är att erbjuda användaren att godkänna eller neka föreslagna maskeringar i ett gransknings-steg, innan man laddar ner filen. Ett annat förslag är att låta användaren granska en "chunk" i taget så att den inte behöver granska hela pdf:en på en gång. Det vore också bra om man kunde välja hur man vill att maskeringen ska se ut, t.ex. om man vill ha ***** eller ett svart streck över orden.

## Insikter
Det svåraste med denna uppgiften är att få rätt predictions från modellen. Även en stor modell som GPT 4.1 gör många fel. Man kan förbättra resultatet genom att finslipa systemprompten, men man måste också inse modellens begränsningar. Den största risken är att modellen helt missar en entitet. Att den sätter fel etikett eller att den felklassificerar något okänsligt som känsligt är ett mindre problem. Därför är recall det viktigaste mätvärdet, viktigare än precision. Att användaren själv får granska och godkänna/neka är ett bra sätt att komma över modellens imperfektioner.

Vi märkte att modellen ofta blir förvirrad kring mejladresser och behöver tydliga instruktioner kring det. Vi experimenterade med tanken att man skulle undvika helt att modellen får se mejladresser och istället maskera dem på förhand med hjälp av RegEX (eftersom mejladresser följer tydliga mönster). Vi utvecklade aldrig en sån lösning, men det är relevant om man ska använda mindre modeller (t.ex. Gemma). Denna metod skulle också spara på tokens.
