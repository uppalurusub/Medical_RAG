import json
from app.implementations.evaluation_implementation import EvaluationImplementation
from app.implementations.rag_implementation import RAGImplementation
if __name__ == "__main__":
    evaluator, rag = EvaluationImplementation(), RAGImplementation()
    result = evaluator.evaluate("./data/eval/rag_eval.jsonl",4,None,lambda q,k:rag.query_for_eval(q,k))
    print(json.dumps(result,indent=2,ensure_ascii=False))
