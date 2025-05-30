
import pandas as pd

def ragas_evaluate(question, answer, contexts):
    try:
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.evaluation import evaluate

        df = pd.DataFrame({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        })

        metrics = [faithfulness, answer_relevancy]
        results = evaluate(df, metrics=metrics)
        # Returnera resultat som dict med floats
        return {
            "faithfulness": float(results["faithfulness"][0]),
            "answer_relevancy": float(results["answer_relevancy"][0])
        }
    except Exception as e:
        return {"error": str(e)}
