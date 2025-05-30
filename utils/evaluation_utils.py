import pandas as pd

def ragas_evaluate(question, answer, contexts):
    try:
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.evaluation import evaluate
        from ragas.data import Dataset

        # Steg 1: Bygg DataFrame
        df = pd.DataFrame({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        })

        # Steg 2: Gör om till RAGAS Dataset
        ragas_dataset = Dataset.from_pandas(df)

        # Steg 3: Utvärdera
        metrics = [faithfulness, answer_relevancy]
        results = evaluate(ragas_dataset, metrics=metrics)

        # Steg 4: Returnera resultat
        return {
            "faithfulness": float(results["faithfulness"][0]),
            "answer_relevancy": float(results["answer_relevancy"][0])
        }
    except Exception as e:
        return {"error": str(e)}
