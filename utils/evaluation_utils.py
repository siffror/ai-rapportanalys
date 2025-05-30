import pandas as pd # Behålls om du använder pandas någon annanstans, annars kan den tas bort från denna fil
import os
from dotenv import load_dotenv

load_dotenv()

from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate, EvaluationDataset
# Importera SingleTurnSample direkt från dess schema-definition
from ragas.dataset.schema import SingleTurnSample # Eller ibland från ragas.llms.object (beroende på RAGAS version)

from langchain_openai import ChatOpenAI

def ragas_evaluate(question: str, answer: str, contexts: list[str]):
    """
    Utvärderar ett givet svar med RAGAS-metriker.
    """
    try:
        # Steg 1: Skapa SingleTurnSample-objektet manuellt
        # Detta är det objekt som RAGAS använder för att representera en enskild "rad" i ditt dataset.
        current_sample = SingleTurnSample(
            question=question,
            answer=answer,
            contexts=contexts
            # ground_truth behövs inte för faithfulness och answer_relevancy här
        )
        
        # Steg 2: Skapa EvaluationDataset från en lista av manuellt skapade samples
        # Eftersom vi bara har ett sample blir det en lista med ett element.
        # Notera: argumentet heter 'samples' här i konstruktorn.
        eval_dataset = EvaluationDataset(samples=[current_sample])

        # Steg 3: Initiera LLM för RAGAS-metriker
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return {"error": "OPENAI_API_KEY hittades inte i miljövariablerna."}
        
        llm_for_ragas = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key, temperature=0)

        # Steg 4: Utvärdera
        metrics_to_evaluate = [faithfulness, answer_relevancy]
        
        result = evaluate(dataset=eval_dataset, metrics=metrics_to_evaluate, llm=llm_for_ragas)

        # Steg 5: Returnera resultat
        faithfulness_score = result.scores['faithfulness'][0] if 'faithfulness' in result.scores else None
        answer_relevancy_score = result.scores['answer_relevancy'][0] if 'answer_relevancy' in result.scores else None

        if faithfulness_score is None or answer_relevancy_score is None:
            missing = []
            if faithfulness_score is None: missing.append("faithfulness")
            if answer_relevancy_score is None: missing.append("answer_relevancy")
            return {"error": f"Kunde inte hämta scores för: {', '.join(missing)}. Fick: {result.scores}"}

        return {
            "faithfulness": float(faithfulness_score),
            "answer_relevancy": float(answer_relevancy_score)
        }

    except ImportError as ie:
        return {"error": f"Importfel i RAGAS: {str(ie)}. Kontrollera installationen."}
    except Exception as e:
        # För felsökning, skriv ut hela traceback för att se exakt var felet uppstod
        import traceback
        print("Detaljerad traceback för RAGAS-fel:")
        print(traceback.format_exc())
        return {"error": f"Generellt fel under RAGAS-utvärdering: {str(e)}"}
