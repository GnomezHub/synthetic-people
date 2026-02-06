
Uppdrag:

Rootpi beställer ett verktyg där användaren väljer en pdf och får tillbaka en maskerad pdf utan personlig information.
Dessa dokument ska kunna användas för till exempel forskning utan att personuppgifterna röjs

Testa:

Först installera Tesseract(text från bilder) och poppler(bild från pdf):

macOS (via Homebrew):
    brew install tesseract poppler

Ubuntu/Linux:
    sudo apt update
    sudo apt install tesseract-ocr poppler-utils

Windows: Ladda ner binärer för Tesseract och Poppler och lägg till dem i din PATH.

sen behöver du installera python biblioteken med detta kommandot:
    pip install flask openai pdfplumber pdf2image pytesseract pillow fpdf werkzeug

och sista steget innan körning är att ställa in openai nyckel:
   
macOS / Linux:
    export OPENAI_API_KEY='[api_nycklen_här]'

Windows (PowerShell):
    $env:OPENAI_API_KEY='[api_nycklen_här]'

sen är de klart och du går till:
    cd flask

och där kör du:
    python app.py

när terminalen säger "running on http..." så öppna http://localhost:5000 från webbläsaren


Kravspecifikation:

Extrahering av information från en PDF fil från användaren
Identifiering av personlig identifierbar information (AI)
Maskering av dessa och generera en PDF fil till användaren

Projektet är tänkt i python och vi använder LLM för identifieringen. RootPi erbjuder API nyckel till openai.

Projektet startades under ledning av en lärare från Lexicon som föreslog att vi börjar med att använda lokal modell, troligtvis av forskningsskäl.
Vi började med Ollama och den enda modellen vi kunde köra på vår dator, men desförinnan behövde vi skapa testdata för våra experiment.
Github projektet fick sitt namn för att spegla den första delen av projektet, och då vid tog fram testdata av påhittade personer så kändes
synthetic people som ett passande namn.

]Ingrid]....testdata, experiment ochmätningar....

Visdomsdelar

Kostnaden är en faktor som kan regleras med välskriven kod, och i LLM sammanhang handlar det om att hushålla med tokens och maxa all logic som går att lägga utanför. För att minska på storleken i svaret så hittade vi ett alternativt sätt till att be om en json list.
Den lokala LLM var usel på att räkna rätt index på dom funna entiteterna så vi flyttade den logiken från modellen till kod. Kvar var det att veta vilken typ av entitet det är och vi valde att ange detta kodat på den första siffran i varje sträng i den listan.
Man kan fortsätta att lägga fler uträkningar som går att göra utan LLM, som email och annat som följer ett mönster som man kanske kan upptäcka med regex. 
Om det förmaskeras så att LLM inte ens behöver tänka på att de finns, så blir det inte bara billigare utan också mer träffsäkert. 

Lösningens komponenter

Vi använde flask att bygga gränsnittet i med html&bootstrap/ javascript. 
inläsningen av PDF. Identifiera informationen och extrahera den en eller flera python bibliotek (fallback).
När vi har informationen ska den delas upp i "chunks" för att inte skicka allting i en och samma fråga till modellen.
Experimenten vi gjorde med ollama visade att modeller kan bli förvirrade av email och kom fram till att man skulle kunna undgå detta
genom att i detta skedet använda regex för att förmaskera email. Dock hade vi inte haft tid att testa det på openapi så istället för att
lägga ännu mer tid på det så ville vi testa utan för att se om det ens behövdes.
Varje chunk går sedan till LLM med instruktioner vi systempromten och kommer tillbaka med upplistade entiteter. 
Vilken typ av entitet den har hittat visar den med första siffran av varje entitet i listan. Sedan hittas index genom att leta igenom entiteten i texten.
Dessa index markerar var i texten som perso infon ska maskeras. I ett gransknings steg visar man dessa förekomster med orden markerade i färgkod för att visa vilken typ av personinfo som hittats
Här kan man utöka UI genom att man i granskningen kan välja och ändra eller ta bort markerade förekomster för maskering. Och man ska kunna bläddra mellan varje "chunk"
så att användaren slipper granska hela pdf dokumentet på en och samma gång. Dessa saker fick av tidskäl bli föremål för uppgraderingar.
 I denna första version kan man bara granska allt den hittat och välja exportera pdf. Då får man pdfn så som man ser den i listan men maskerad med [förekomst], till exempel [telefonnummer].
Man skulle också innan exportknappen kunna ha valet av maskering, som [förekomstnamn], med **** eller med svart box över orden. Här blev det [förekomstnamn].

Övriga förbättringar

I extraktionen istället för pdfplumber använda PyMuPDF (för att få ut koordinater/bbox).
Analysen behöver inte bara läggas på LLM utan en hybrid: Presidio (Regex + Spacy) -> LLM (för kontext).
För exporten använda PyMuPDF (ritar rutor på original-PDF).
UIX mer interaktion i gransknings och export bitarna, som jag beskrev ovan i "lösningens komponenter".
