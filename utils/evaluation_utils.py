# utils/evaluation_utils.py (Anpassad för RAGAS 0.2.15 - med Hugging Face Dataset)

import os
from dotenv import load_dotenv

# RAGAS Importer för v0.2.x
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate

from langchain_openai import ChatOpenAI

# Importera Dataset från Hugging Face datasets-biblioteket
from datasets import Dataset # <--- NY IMPORT

load_dotenv()

def ragas_evaluate(question: str, answer: str, contexts: list[str]):
    """
    Utvärderar ett givet svar med RAGAS-metriker (anpassad för RAGAS v0.2.x med HF Dataset).
    """
    print(f"[RAGAS v0.2.x Debug] ragas_evaluate anropad. Fråga (user_input): '{question[:30]}...'")
    try:
        # Steg 1: Förbered data som en lista av dictionaries (samma som tidigare)
        dataset_list_of_dicts = [
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts
                # "ground_truth": "..." # Lägg till om du har referenssvar
            }
        ]
        print(f"[RAGAS v0.2.x Debug] Python list of dicts förberett: {dataset_list_of_dicts}")

        # Steg 1.5: Konvertera listan av dictionaries till ett Hugging Face Dataset-objekt
        # Detta är ofta det föredragna och mest robusta formatet för RAGAS 'evaluate'-funktion.
        hf_dataset = Dataset.from_list(dataset_list_of_dicts)
        print(f"[RAGAS v0.2.x Debug] Hugging Face Dataset skapat: {hf_dataset}")


        # Steg 2: Initiera LLM för RAGAS-metriker
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("[RAGAS v0.2.x Debug] Fel: OPENAI_API_KEY saknas.")
            return {"error": "OPENAI_API_KEY hittades inte i miljövariablerna."}
        
        llm_for_ragas = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0)
        print("[RAGAS v0.2.x Debug] LLM för RAGAS initierad.")

        # Steg 3: Definiera metriker och kör utvärderingen
        metrics_to_evaluate = [faithfulness, answer_relevancy]
        print(f"[RAGAS v0.2.x Debug] Kör utvärdering med metriker: {metrics_to_evaluate}")
        
        # Skicka nu Hugging Face Dataset-objektet till 'dataset'-argumentet
        result = evaluate(
            dataset=hf_dataset, 
            metrics=metrics_to_evaluate, 
            llm=llm_for_ragas
        )
        
        print(f"[RAGAS v0.2.x Debug] Utvärdering klar. Resultattyp: {type(result)}")

        # Steg 4: Extrahera och returnera resultat
        # När input är ett Hugging Face Dataset, är output (result) vanligtvis också ett 
        # Hugging Face Dataset som innehåller de ursprungliga kolumnerna plus de nya score-kolumnerna.
        if hasattr(result, 'to_pandas'): 
             result_df = result.to_pandas()
             print(f"[RAGAS v0.2.x Debug] Resultat (som DataFrame): \n{result_df}")
             # För ett enskilt sample i en DataFrame
             faithfulness_score = result_df["faithfulness"].iloc[0] if "faithfulness" in result_df.columns and not result_df.empty else None
             answer_relevancy_score = result_df["answer_relevancy"].iloc[0] if "answer_relevancy" in result_df.columns and not result_df.empty else None
        elif isinstance(result, dict): # Fallback om resultatet är en direkt dictionary
            print(f"[RAGAS v0.2.x Debug] Resultat (som dict): {result}")
            faithfulness_score = result.get("faithfulness")
            answer_relevancy_score = result.get("answer_relevancy")
        else: 
            print(f"[RAGAS v0.2.x Debug] Oväntad resultattyp. Försöker komma åt 'scores'-attribut ifall det är RAGAS 1.0+ stil.")
            # För RAGAS 1.0+ är result.scores en dict. Detta är en gissning om 0.2.15 skulle ha en liknande struktur.
            faithfulness_score = result.scores.get("faithfulness") if hasattr(result, 'scores') and isinstance(result.scores, dict) else None
            answer_relevancy_score = result.scores.get("answer_relevancy") if hasattr(result, 'scores') and isinstance(result.scores, dict) else None

        if faithfulness_score is None or answer_relevancy_score is None:
            missing = []
            if faithfulness_score is None: missing.append("faithfulness")
            if answer_relevancy_score is None: missing.append("answer_relevancy")
            error_msg = f"Kunde inte hämta alla scores för: {', '.join(missing)}. Mottaget resultat: {result}"
            print(f"[RAGAS v0.2.x Debug] Fel vid hämtning av scores: {error_msg}")
            return {"error": error_msg}

        print(f"[RAGAS v0.2.x Debug] Faithfulness: {faithfulness_score}, Answer Relevancy: {answer_relevancy_score}")
        return {
            "faithfulness": float(faithfulness_score),
            "answer_relevancy": float(answer_relevancy_score)
        }

    except Exception as e:
        import traceback
        detailed_error = traceback.format_exc()
        print(f"[RAGAS v0.2.x Debug] Detaljerad traceback för RAGAS-fel:\n{detailed_error}")
        return {"error": f"Generellt fel under RAGAS-utvärdering (v0.2.15): {str(e)}"}
