def _int(value):
    try: return int(value) if value is not None else None
    except (TypeError, ValueError): return None

def extract_usage(message):
    """Extract actual provider usage from the AIMessage; do not fabricate missing usage."""
    response_meta = getattr(message, "response_metadata", None) or {}
    usage_meta = getattr(message, "usage_metadata", None) or {}
    candidates = [usage_meta, response_meta.get("token_usage"), response_meta.get("usage"),
                  response_meta.get("tokenUsage")]
    usage = next((x for x in candidates if isinstance(x, dict) and x), {})
    inp = _int(usage.get("input_tokens", usage.get("prompt_tokens", usage.get("promptTokenCount"))))
    out = _int(usage.get("output_tokens", usage.get("completion_tokens", usage.get("candidatesTokenCount"))))
    total = _int(usage.get("total_tokens", usage.get("totalTokens")))
    if total is None and inp is not None and out is not None: total = inp + out
    raw_cost = usage.get("cost", response_meta.get("cost"))
    try: cost = float(raw_cost) if raw_cost is not None else None
    except (TypeError, ValueError): cost = None
    return {"input_tokens":inp,"output_tokens":out,"total_tokens":total,
            "input_cost_usd":None,"output_cost_usd":None,"total_cost_usd":cost,
            "source":"provider_usage" if usage else "unavailable"}
