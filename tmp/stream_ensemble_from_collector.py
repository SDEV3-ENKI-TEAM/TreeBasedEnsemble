import json, time, signal, sys, os
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import joblib
from confluent_kafka import Consumer, Producer, KafkaError, KafkaException

# ── 설정 ───────────────────────────
BOOTSTRAP = "13.125.78.213:19092"
IN_TOPIC  = "raw_trace"         #kafka 트레이스 받는 토픽
OUT_TOPIC = "ensemble_predict"  #kafka 결과 전송 토픽
GROUP_ID  = "ensemble-predictor" 

MODEL_DIR = "./model_ens_out"    
THRESHOLD = 0.75               

from trace_features import TraceFeaturizer

# ========== Kafka 유틸 ==========
def build_consumer() -> Consumer:
    return Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "max.poll.interval.ms": 300000,
        "session.timeout.ms": 45000,
    })

def build_producer() -> Producer:
    return Producer({"bootstrap.servers": BOOTSTRAP})

# ========== OTLP(JSON) → TraceFeaturizer 입력형식 ==========
def _val_from_any(value_obj: Dict[str, Any]):
    if not isinstance(value_obj, dict):
        return value_obj
    for k in ("stringValue", "intValue", "boolValue", "doubleValue"):
        if k in value_obj:
            return value_obj[k]
    if "arrayValue" in value_obj:
        arr = value_obj["arrayValue"].get("values", [])
        out = []
        for x in arr:
            v = _val_from_any(x)
            if v is not None:
                out.append(v)
        return out
    if "bytesValue" in value_obj:
        return value_obj["bytesValue"]
    return None

def otlp_span_to_jaeger_like(span: Dict[str, Any]) -> Dict[str, Any]:
    st = span.get("startTimeUnixNano")
    try:
        start_ns = int(st) if st is not None else None
    except Exception:
        start_ns = None

    tags = []
    for kv in span.get("attributes", []):
        k = kv.get("key")
        v = _val_from_any(kv.get("value", {}))
        if isinstance(v, list):
            for i, iv in enumerate(v):
                tags.append({"key": f"{k}[{i}]", "value": iv})
        else:
            tags.append({"key": k, "value": v})

    if "name" in span:
        tags.append({"key": "SpanName", "value": span["name"]})
    if "traceId" in span:
        tags.append({"key": "traceId", "value": span["traceId"]})
    if "spanId" in span:
        tags.append({"key": "spanId", "value": span["spanId"]})

    return {"startTime": start_ns, "tags": tags}

def otlp_to_trace_json(otlp_json: Dict[str, Any]) -> Dict[str, Any]:
    spans_out, trace_id = [], None
    for rs in otlp_json.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for sp in ss.get("spans", []):
                if trace_id is None:
                    trace_id = sp.get("traceId")
                spans_out.append(otlp_span_to_jaeger_like(sp))
    return {"traceID": trace_id, "spans": spans_out}

# ========== 모델 로딩 & 스코어링 ==========
def load_models(model_dir: str):
    folds = joblib.load(os.path.join(model_dir, "ensemble_models.joblib"))
    with open(os.path.join(model_dir, "feature_columns.json"), "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    return folds, feature_columns

def build_matrix_one(df_row: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    feats = df_row.reindex(columns=feature_columns, fill_value=0.0)
    feats = feats.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feats = feats.astype("float64")
    return feats

def score_trace(featurizer, folds, feature_columns, trace_json: dict) -> dict:
    df_row = pd.DataFrame([featurizer.featurize_trace(trace_json)])  
    Xdf   = build_matrix_one(df_row, feature_columns)            

    p_l = np.mean([fd["lgbm"].predict_proba(Xdf)[:, 1] for fd in folds]).item()
    p_x = np.mean([fd["xgb" ].predict_proba(Xdf)[:, 1] for fd in folds]).item()
    p_c = np.mean([fd["cat" ].predict_proba(Xdf)[:, 1] for fd in folds]).item()
    p_m = (p_l + p_x + p_c) / 3.0

    return {
        "score_lgbm": float(p_l),
        "score_xgb":  float(p_x),
        "score_cat":  float(p_c),
        "score_mean": float(p_m),
        "prediction": "malicious" if p_m >= THRESHOLD else "benign",
        "threshold": THRESHOLD,
    }

# ========== 메인 루프 ==========
RUNNING = True
def _sig(_s, _f):
    global RUNNING
    RUNNING = False

def delivery_report(err, _msg):
    if err is not None:
        sys.stderr.write(f"[Kafka] delivery failed: {err}\n")

def main():
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # 모델 & 피처
    folds, feature_columns = load_models(MODEL_DIR)    
    tf = TraceFeaturizer()                             

    # Kafka
    consumer = build_consumer()
    producer = build_producer()
    consumer.subscribe([IN_TOPIC])

    print(f"[+] consume: {IN_TOPIC}  ->  produce: {OUT_TOPIC}")
    try:
        while RUNNING:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    raise KafkaException(msg.error())
                continue

            try:
                payload = msg.value()
                if not payload:
                    continue
                otlp = json.loads(payload.decode("utf-8"))  # encoding: otlp_json
            except Exception as e:
                sys.stderr.write(f"[parse] invalid JSON: {e}\n")
                continue

            trace = otlp_to_trace_json(otlp)  # OTLP → Featurizer 입력형식
            trace_id = trace.get("traceID") or ""

            try:
                result = score_trace(tf, folds, feature_columns, trace)
            except Exception as e:
                sys.stderr.write(f"[score] failed trace({trace_id}): {e}\n")
                continue

            out_msg = {
                "traceID": trace_id,
                "score": result["score_mean"],
                "prediction": result["prediction"]
            }

            try:
                producer.produce(
                    topic=OUT_TOPIC,
                    key=str(trace_id),
                    value=json.dumps(out_msg, ensure_ascii=False).encode("utf-8"),
                    callback=delivery_report
                )
                producer.poll(0)
            except BufferError:
                producer.flush()
    finally:
        try: consumer.close()
        except: pass
        try: producer.flush(5)
        except: pass
        print("[+] shutdown")

if __name__ == "__main__":
    main()
