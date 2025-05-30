import pandas as pd
import os # För att hämta API-nyckel
from dotenv import load_dotenv # För att ladda .env-filen om det behövs här

# Ladda miljövariabler om .env-fil finns i samma mapp eller en överordnad mapp
# Om app.py redan kör load_dotenv() kanske detta inte är strikt nödvändigt här,
# men det skadar inte att säkerställa att API-nyckeln är tillgänglig.
load_dotenv()

# RAGAS Importer - Uppdaterade
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate # Ändrad från ragas.evaluation
from ragas import EvaluationDataset # Ny import för det nya dataset-objektet

# Importera LLM-modellen du vill använda (t.ex. ChatOpenAI)
from langchain_openai import ChatOpenAI

def ragas_evaluate(question: str, answer: str, contexts: list[str]):
    """
    Utvärderar ett givet svar med RAGAS-metriker.

    Args:
        question (str): Frågan som ställdes.
        answer (str): Svaret som genererades av LLM.
        contexts (list[str]): En lista av kontext-strängar som användes för att generera svaret.

    Returns:
        dict: En dictionary med metriker eller ett felmeddelande.
    """
    try:
        # Steg 1: Förbered data för EvaluationDataset
        # RAGAS EvaluationDataset förväntar sig att varje fält är en lista.
        # Eftersom vi utvärderar ett enskilt sample (fråga-svar-par),
        # skapar vi listor med ett element vardera.
        data_samples = {
            'question': [question],
            'answer': [answer],
            'contexts': [contexts]  # 'contexts' är redan en lista av strängar, så vi lägger den i en yttre lista.
        }
        
        # Skapa EvaluationDataset-objektet
        eval_dataset = EvaluationDataset.from_dict(data_samples)

        # Steg 2: Initiera LLM för RAGAS-metriker
        # Faithfulness och AnswerRelevancy kräver en LLM.
        # Hämta API-nyckeln från miljövariabler
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return {"error": "OPENAI_API_KEY hittades inte i miljövariablerna."}
        
        # Du kan välja vilken modell du vill använda här, t.ex. "gpt-3.5-turbo" eller "gpt-4o"
        llm_for_ragas = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key, temperature=0)

        # Steg 3: Utvärdera
        # Definiera metriker
        metrics_to_evaluate = [faithfulness, answer_relevancy]
        
        # Kör utvärderingen
        # Notera att vi skickar med 'llm' argumentet nu.
        result = evaluate(dataset=eval_dataset, metrics=metrics_to_evaluate, llm=llm_for_ragas)

        # Steg 4: Returnera resultat
        # 'result' är ett Result-objekt. Scores finns i 'result.scores'.
        # 'result.scores' är ett Hugging Face Dataset-objekt.
        # Därför behöver vi indexera för att få värdena för vårt enda sample.
        faithfulness_score = result.scores['faithfulness'][0] if 'faithfulness' in result.scores else None
        answer_relevancy_score = result.scores['answer_relevancy'][0] if 'answer_relevancy' in result.scores else None

        if faithfulness_score is None or answer_relevancy_score is None:
            return {"error": f"Kunde inte hämta alla scores. Fick: {result.scores}"}

        return {
            "faithfulness": float(faithfulness_score),
            "answer_relevancy": float(answer_relevancy_score)
        }

    except ImportError as ie:
        # Specifik felhantering för importfel kan vara bra under utveckling
        return {"error": f"Importfel i RAGAS: {str(ie)}. Se till att RAGAS är korrekt installerat och att alla komponenter är tillgängliga."}
    except Exception as e:
        # Generell felhantering
        # Du kan logga det fullständiga felet för felsökning om du vill
        # import traceback
        # print(traceback.format_exc())
        return {"error": f"Generellt fel under RAGAS-utvärdering: {str(e)}"}
