
import json
import re
import os
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

# --------- Tokenizers ---------
_PATH_SPLIT = re.compile(r"[\\/]+")

def _path_tokens(p: str):
    toks = []
    for seg in _PATH_SPLIT.split(p.lower()):
        if not seg:
            continue
        toks.append(seg)
        if "." in seg:
            name, ext = seg.rsplit(".", 1)
            toks.append("ext:"+ext)
    return toks

def _domain_tokens(d: str):
    d = d.lower()
    toks = d.split(".")
    if len(toks) >= 1:
        toks.append("tld:"+toks[-1])
    toks.append("len:"+str(len(d)))
    digits = sum(c.isdigit() for c in d)
    toks.append("digit_ratio:"+str(digits/max(1,len(d))))
    return toks

def _parse_time_ns(span):
    # Prefer Jaeger startTime (ns). Fall back to UtcTime string if needed.
    st = span.get("startTime")
    if st is not None:
        try:
            return int(st)
        except Exception:
            pass
    # Fallback: parse tags.UtcTime like "2025-08-29 19:29:37.337"
    for t in span.get("tags", []):
        if t.get("key") == "UtcTime":
            s = t.get("value", "")
            # Try different formats
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(s, fmt)
                    # convert to ns epoch (assume naive UTC)
                    epoch = datetime(1970,1,1)
                    return int((dt - epoch).total_seconds() * 1e9)
                except Exception:
                    continue
    return None

def _tags_to_dict(tags):
    d = {}
    for t in tags:
        d[t.get("key")] = t.get("value")
    return d

# --------- Core featurizer ---------
class TraceFeaturizer:
    def __init__(self, n_hash_path: int = 2048, n_hash_dom: int = 1024):
        self.n_hash_path = n_hash_path
        self.n_hash_dom = n_hash_dom
        self._hasher_path = FeatureHasher(n_features=self.n_hash_path, input_type='string')
        self._hasher_dom  = FeatureHasher(n_features=self.n_hash_dom,  input_type='string')

    def featurize_trace(self, trace_json: dict) -> dict:
        spans = trace_json.get("spans", [])
        counts = Counter()
        path_tok_bag = []
        dom_tok_bag  = []
        times_ns = []
        proc_names = Counter()
        parents = Counter()

        for s in spans:
            tags_d = _tags_to_dict(s.get("tags", []))
            # Event basics
            evt_id_raw = tags_d.get("ID")
            try:
                evt_id = int(evt_id_raw)
            except Exception:
                evt_id = -1
            evt_nm = tags_d.get("EventName", "")
            rule   = tags_d.get("RuleName", "-")

            counts[f"evt_id:{evt_id}"] += 1
            if evt_nm:
                counts[f"evt_nm:{evt_nm}"] += 1
            if rule:
                counts[f"rule:{rule}"] += 1

            # Process / parent
            img = tags_d.get("Image", "")
            if img:
                fname = _PATH_SPLIT.split(img.lower())[-1]
                proc_names[fname] += 1
                for tk in _path_tokens(img):
                    path_tok_bag.append("img:"+tk)

            ppid = tags_d.get("sysmon.ppid")
            if ppid is not None:
                parents[str(ppid)] += 1

            # File path features
            tf = tags_d.get("TargetFilename", "")
            if tf:
                for tk in _path_tokens(tf):
                    path_tok_bag.append("tf:"+tk)

            # DNS features
            qn = tags_d.get("QueryName", "")
            if qn:
                for tk in _domain_tokens(qn):
                    dom_tok_bag.append("qn:"+tk)
            qs = tags_d.get("QueryStatus", "")
            if qs:
                counts[f"qstatus:{qs}"] += 1

            # time
            tns = _parse_time_ns(s)
            if tns is not None:
                times_ns.append(tns)

        # Time statistics
        times_ns.sort()
        if len(times_ns) >= 2:
            dur = times_ns[-1] - times_ns[0]
            gaps = np.diff(times_ns)
        else:
            dur = 0
            gaps = np.array([0])

        time_feats = {
            "trace_dur_ns": float(dur),
            "gap_mean": float(np.mean(gaps)),
            "gap_std": float(np.std(gaps)),
            "gap_p90": float(np.quantile(gaps,0.9)) if gaps.size>0 else 0.0,
            "gap_burst": float((np.quantile(gaps,0.9)+1.0)/(np.median(gaps)+1.0)) if gaps.size>0 else 1.0
        }

        dense = {
            "evt_count": float(sum(v for k,v in counts.items() if k.startswith("evt_id:"))),
            "uniq_proc": float(len(proc_names)),
            "uniq_parent": float(len(parents)),
            "top_proc_1": float(proc_names.most_common(1)[0][1]) if proc_names else 0.0,
            "top_proc_2": float(proc_names.most_common(2)[-1][1]) if len(proc_names)>=2 else 0.0,
            "top_proc_3": float(proc_names.most_common(3)[-1][1]) if len(proc_names)>=3 else 0.0,
        }
        dense.update(time_feats)

        # Convert counters to flat dict
        df_counts = pd.Series(counts, dtype=float)

        feat = pd.concat([pd.Series(dense, dtype=float), df_counts], axis=0).to_dict()

        # Hash bags
        if path_tok_bag:
            Xp = self._hasher_path.transform([[tok] for tok in path_tok_bag]).toarray().sum(axis=0)
        else:
            Xp = np.zeros(self.n_hash_path, dtype=float)
        if dom_tok_bag:
            Xd = self._hasher_dom.transform([[tok] for tok in dom_tok_bag]).toarray().sum(axis=0)
        else:
            Xd = np.zeros(self.n_hash_dom, dtype=float)

        for i, v in enumerate(Xp):
            feat[f"hash_path_{i}"] = float(v)
        for i, v in enumerate(Xd):
            feat[f"hash_dom_{i}"]  = float(v)

        feat["traceID"] = trace_json.get("traceID")
        return feat

    def featurize_dir(self, data_dir: str, pattern: str = "*.json") -> pd.DataFrame:
        rows = []
        import glob
        for fp in glob.glob(os.path.join(data_dir, pattern)):
            with open(fp, "r", encoding="utf-8-sig") as f:
                j = json.load(f)
            rows.append(self.featurize_trace(j))
        df = pd.DataFrame(rows).fillna(0.0)
        return df
