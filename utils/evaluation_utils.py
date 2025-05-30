
def simple_evaluate(answer, required_keywords=None, min_length=30):
    score = 1.0
    feedback = []
    if len(answer) < min_length:
        feedback.append("⚠️ Svaret är väldigt kort.")
        score -= 0.5
    if required_keywords:
        for word in required_keywords:
            if word.lower() not in answer.lower():
                feedback.append(f"⚠️ Saknar nyckelord: {word}")
                score -= 0.2
    return max(0, score), feedback

# Om du vill lägga till fler utvärderingar kan du göra det här!
# utils/evaluation_utils.py

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
