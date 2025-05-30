# utils/evaluation_utils.py

import os
from dotenv import load_dotenv

# RAGAS Importer - Försöker hitta SingleTurnSample på olika ställen
print("[RAGAS Debug] Försöker importera SingleTurnSample...")
try:
    from ragas.dataset.schema import SingleTurnSample # Försök 1 (Mest korrekta för RAGAS 0.1.9)
    print("[RAGAS Debug] Importerade SingleTurnSample från ragas.dataset.schema")
except (ImportError, ModuleNotFoundError) as e1:
    print(f"[RAGAS Debug] Misslyckades importera från ragas.dataset.schema: {e1}")
    try:
        from ragas import SingleTurnSample # Försök 2 (Om den exporteras på toppnivå)
        print("[RAGAS Debug] Importerade SingleTurnSample från ragas (toppnivå)")
    except (ImportError, ModuleNotFoundError) as e2:
        print(f"[RAGAS Debug] Misslyckades importera från ragas (toppnivå): {e2}")
        try:
            from ragas.llms.object import SingleTurnSample # Försök 3 (Äldre/alternativ sökväg)
            print("[RAGAS Debug] Importerade SingleTurnSample från ragas.llms.object")
        except (ImportError, ModuleNotFoundError) as e3:
            print(f"[RAGAS Debug] Misslyckades importera från ragas.llms.object: {e3}")
            # Om alla försök misslyckas, lyft ett tydligt fel som kan ses i Streamlit-loggarna
            # och som kommer att stopppa appen från att försöka använda en saknad komponent.
            raise ImportError(
                "Kunde inte importera SingleTurnSample från kända RAGAS-sökvägar "
                "(ragas.dataset.schema, ragas, ragas.llms.object). "
                "Kontrollera RAGAS installation (förväntar sig v0.1.9) och Streamlit Cloud loggar. "
                f"Fel detaljer: SchemaPath='{e1}', TopLevelPath='{e2}', LLMObjectPath='{e3}'"
            )

# Fortsätt med andra nödvändiga importer
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate, EvaluationDataset # EvaluationDataset behövs fortfarande
from langchain_openai import ChatOpenAI

load_dotenv()

def ragas_evaluate(question: str, answer: str, contexts: list[str]):
    """
    Utvärderar ett givet svar med RAGAS-metriker.
    """
    print(f"[RAGAS Debug] ragas_evaluate anropad med fråga: '{question[:30]}...'")
    try:
        # Steg 1: Skapa SingleTurnSample-objektet manuellt
        # Detta använder den 'SingleTurnSample' som lyckades importeras ovan.
        current_sample = SingleTurnSample(
            question=question,
            answer=answer,
            contexts=contexts
        )
        print("[RAGAS Debug] SingleTurnSample-objekt skapat.")
        
        # Steg 2: Skapa EvaluationDataset från en lista av manuellt skapade samples
        eval_dataset = EvaluationDataset(samples=[current_sample])
        print("[RAGAS Debug] EvaluationDataset-objekt skapat.")

        # Steg 3: Initiera LLM för RAGAS-metriker
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("[RAGAS Debug] Fel: OPENAI_API_KEY saknas.")
            return {"error": "OPENAI_API_KEY hittades inte i miljövariablerna."}
        
        llm_for_ragas = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key, temperature=0)
        print("[RAGAS Debug] LLM för RAGAS initierad.")

        # Steg 4: Utvärdera
        metrics_to_evaluate = [faithfulness, answer_relevancy]
        print(f"[RAGAS Debug] Kör utvärdering med metriker: {metrics_to_evaluate}")
        
        result = evaluate(dataset=eval_dataset, metrics=metrics_to_evaluate, llm=llm_for_ragas)
        print(f"[RAGAS Debug] Utvärdering klar. Resultatobjekt: {type(result)}")
        if hasattr(result, 'scores'):
            print(f"[RAGAS Debug] Scores från resultat: {result.scores}")
        else:
            print(f"[RAGAS Debug] Resultatobjektet saknar 'scores'-attribut. Resultat: {result}")


        # Steg 5: Returnera resultat
        faithfulness_score = result.scores['faithfulness'][0] if hasattr(result, 'scores') and 'faithfulness' in result.scores else None
        answer_relevancy_score = result.scores['answer_relevancy'][0] if hasattr(result, 'scores') and 'answer_relevancy' in result.scores else None

        if faithfulness_score is None or answer_relevancy_score is None:
            missing = []
            if faithfulness_score is None: missing.append("faithfulness")
            if answer_relevancy_score is None: missing.append("answer_relevancy")
            error_msg = f"Kunde inte hämta alla scores för: {', '.join(missing)}. Fick: {result.scores if hasattr(result, 'scores') else 'Inga scores tillgängliga'}"
            print(f"[RAGAS Debug] Fel vid hämtning av scores: {error_msg}")
            return {"error": error_msg}

        print(f"[RAGAS Debug] Faithfulness: {faithfulness_score}, Answer Relevancy: {answer_relevancy_score}")
        return {
            "faithfulness": float(faithfulness_score),
            "answer_relevancy": float(answer_relevancy_score)
        }

    except ImportError as ie:
        # Detta fångar om 'raise ImportError' ovan exekveras.
        error_message = f"Nödvändig RAGAS-komponent (SingleTurnSample) kunde inte importeras: {str(ie)}"
        print(f"[RAGAS Debug] {error_message}")
        return {"error": error_message}
    except Exception as e:
        import traceback
        detailed_error = traceback.format_exc()
        print(f"[RAGAS Debug] Detaljerad traceback för RAGAS-fel:\n{detailed_error}")
        return {"error": f"Generellt fel under RAGAS-utvärdering: {str(e)} (se Streamlit-loggar för detaljer)"}
