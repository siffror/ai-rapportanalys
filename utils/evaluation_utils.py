import pandas as pd

def ragas_evaluate(question, answer, contexts):
    try:
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.evaluation import evaluate
        from ragas.data import Dataset

        df = pd.DataFrame({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        })

        ragas_dataset = Dataset.from_pandas(df)
        metrics = [faithfulness, answer_relevancy]
        results = evaluate(ragas_dataset, metrics=metrics)

        return {
            "faithfulness": float(results["faithfulness"][0]),
            "answer_relevancy": float(results["answer_relevancy"][0])
        }
    except Exception as e:
        return {"error": str(e)}
