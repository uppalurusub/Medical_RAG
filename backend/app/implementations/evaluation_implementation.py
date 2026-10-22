import json, re
from pathlib import Path
from collections import Counter
from app.config.config import settings
from app.observability.mlflow_logger import log_metrics

def words(text): return re.findall(r"\\b[\\w'-]+\\b",(text or "").lower())
def token_f1(prediction, reference):
    p,r=words(prediction),words(reference)
    if not p or not r: return 0.0
    overlap=sum((Counter(p)&Counter(r)).values())
    if not overlap: return 0.0
    precision,recall=overlap/len(p),overlap/len(r)
    return 2*precision*recall/(precision+recall)

class EvaluationImplementation:
    def evaluate(self,dataset_path,top_k,max_examples,query_fn):
        path=Path(dataset_path)
        if not path.is_absolute(): path=(Path.cwd()/path).resolve()
        if not path.is_file(): raise ValueError(f"Evaluation dataset not found: {path}")
        with path.open(encoding="utf-8") as f: rows=[json.loads(x) for x in f if x.strip()]
        if max_examples: rows=rows[:max_examples]
        if not rows: raise ValueError("Evaluation dataset is empty")
        hits,precisions,recalls,rrs,f1s,lats=[],[],[],[],[],[]
        details=[]
        for row in rows:
            result=query_fn(row["question"],top_k)
            retrieved=result.get("retrieved_source_ids",[])[:top_k]
            relevant=set(row.get("relevant_source_ids",[]))
            found=[s for s in retrieved if s in relevant]
            hit=float(bool(found)); precision=len(found)/max(len(retrieved),1)
            recall=len(found)/len(relevant) if relevant else None
            rank=next((i+1 for i,s in enumerate(retrieved) if s in relevant),None)
            rr=1/rank if rank else 0.0
            ref=row.get("reference_answer","")
            f1=token_f1(result.get("answer",""),ref) if ref else None
            latency=float(result.get("latency_seconds",0))
            hits.append(hit); precisions.append(precision); rrs.append(rr); lats.append(latency)
            if recall is not None: recalls.append(recall)
            if f1 is not None: f1s.append(f1)
            details.append({"question":row["question"],"retrieved_source_ids":retrieved,
                "relevant_source_ids":sorted(relevant),"hit":hit,"precision_at_k":precision,
                "recall_at_k":recall,"reciprocal_rank":rr,"answer_token_f1":f1,
                "latency_seconds":latency,"answer":result.get("answer","")})
        mean=lambda xs:round(sum(xs)/len(xs),4) if xs else None
        metrics={"hit_rate_at_k":mean(hits),"precision_at_k":mean(precisions),"recall_at_k":mean(recalls),
                 "mrr":mean(rrs),"answer_token_f1":mean(f1s),"mean_latency_seconds":mean(lats),
                 "examples_evaluated":len(details),"top_k":top_k}
        run_id=log_metrics({k:v for k,v in metrics.items() if isinstance(v,(int,float))},
            {"dataset":path.name,"top_k":top_k},{"operation":"rag_evaluation","model":settings.resolved_model or "unknown"})
        return {"status":"completed","examples_evaluated":len(details),"metrics":metrics,"run_id":run_id,"details":details}
