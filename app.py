"""
AI Trust Meter – Streamlit App

Run locally:
  1) Install dependencies (ideally in a virtualenv):
     pip install streamlit sentence-transformers scikit-learn numpy pandas matplotlib textstat language-tool-python

  2) Start the app:
     streamlit run app.py

Notes:
- This app runs offline and uses local models only. If optional models/resources are
  not available, the app degrades gracefully and shows notices in the UI.
- You can extend the local mini-knowledge base in the `LOCAL_KNOWLEDGE_BASE` dictionary.
- No external API keys or HTTP calls are required. Optional components (e.g.,
  language_tool_python) may try to download a local server on first run; if you
  are fully offline and it's not available, grammar checks will be skipped.
"""

from __future__ import annotations

import re
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import streamlit as st

# Heavy libs: use cache_resource for model loading
@st.cache_resource(show_spinner=True)
def _load_models():
    embedder = None
    cross_encoder = None
    embedder_error = None
    cross_error = None
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:  # pragma: no cover
        embedder_error = str(exc)

    # Optional: CrossEncoder for improved coherence scoring if available
    try:
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder("cross-encoder/stsb-TinyBERT-L-4")
    except Exception as exc:  # pragma: no cover
        cross_error = str(exc)

    return embedder, cross_encoder, embedder_error, cross_error


@st.cache_resource(show_spinner=False)
def _load_language_tool():
    try:
        import language_tool_python
        tool = language_tool_python.LanguageTool("en-US")
        return tool, None
    except Exception as exc:  # pragma: no cover
        return None, str(exc)


# -----------------------------
# Local mini knowledge base
# -----------------------------
LOCAL_KNOWLEDGE_BASE: Dict[str, str] = {
    "capital of france": "paris",
    "capital of germany": "berlin",
    "capital of italy": "rome",
    "largest planet": "jupiter",
    "pi approx": "3.14159",
    "python creator": "guido van rossum",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def extract_fact_query(question: str) -> Optional[str]:
    q = normalize_text(question)
    # Simple heuristics to map question to KB key
    patterns = [
        (r"what is the capital of ([a-z\s]+)\??", lambda m: f"capital of {m.group(1).strip()}"),
        (r"capital of ([a-z\s]+)\??", lambda m: f"capital of {m.group(1).strip()}"),
        (r"largest planet\??", lambda m: "largest planet"),
        (r"who created python\??", lambda m: "python creator"),
        (r"approx(?:imation)? of pi\??", lambda m: "pi approx"),
    ]
    for pat, fn in patterns:
        m = re.search(pat, q)
        if m:
            return normalize_text(fn(m))
    # Fallback: allow direct KB key presence
    for key in LOCAL_KNOWLEDGE_BASE.keys():
        if key in q:
            return key
    return None


def simple_fact_check(question: str, answer: str) -> Tuple[float, str]:
    """Return factuality in [0,1] and rationale string.
    Uses the local KB if a key can be extracted, otherwise heuristic checks.
    """
    q_key = extract_fact_query(question)
    ans_norm = normalize_text(answer)
    if q_key and q_key in LOCAL_KNOWLEDGE_BASE:
        expected = normalize_text(LOCAL_KNOWLEDGE_BASE[q_key])
        # direct containment or exact match
        if expected in ans_norm:
            return 1.0, f"Matched KB: {q_key} -> {expected}"
        else:
            # Penalize if answer mentions a conflicting known entity (e.g., other capitals)
            conflicts = [v for k, v in LOCAL_KNOWLEDGE_BASE.items() if v != LOCAL_KNOWLEDGE_BASE[q_key] and v in ans_norm]
            if conflicts:
                return 0.0, f"Conflict with KB: expected '{expected}', found '{conflicts[0]}'"
            return 0.4, f"Did not contain expected KB value '{expected}'"

    # Heuristic numeric consistency: if Q mentions a year/number, prefer same in A
    q_nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", question)
    a_nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", answer)
    if q_nums:
        factual = float(len(set(q_nums).intersection(set(a_nums))) > 0)
        rationale = "Number overlap" if factual == 1.0 else "No numeric overlap"
        return factual, rationale

    # Default uncertain factuality when we cannot check
    return 0.6, "No KB match; heuristic neutral score"


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    denom = (np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0:
        return 0.0
    return float(np.dot(u, v) / denom)


def compute_coherence(embedder, cross_encoder, question: str, answer: str) -> Tuple[float, str]:
    """Compute semantic coherence [0,1] using embeddings or optional cross-encoder."""
    q = question.strip()
    a = answer.strip()
    if not q or not a or embedder is None:
        return 0.5, "Embedder unavailable or empty text"

    try:
        q_emb = embedder.encode([q])[0]
        a_emb = embedder.encode([a])[0]
        cos = cosine_similarity(q_emb, a_emb)
        # Map cosine [-1,1] to [0,1]
        sim = max(0.0, min(1.0, (cos + 1.0) / 2.0))
        rationale = f"Cosine={cos:.3f}"
        # Optional refinement with cross-encoder, if available
        if cross_encoder is not None:
            try:
                score = float(cross_encoder.predict([(q, a)])[0])
                # CrossEncoder STS scores often in [0, 5] or [0,1] depending on model; normalize conservatively
                if score > 1.5:  # likely STS [0,5]
                    score = score / 5.0
                sim = 0.5 * sim + 0.5 * max(0.0, min(1.0, score))
                rationale += f", Cross={score:.3f}"
            except Exception:
                pass
        return sim, rationale
    except Exception as exc:  # pragma: no cover
        return 0.5, f"Embedding error: {exc}"


def compute_clarity(answer: str, lt_tool) -> Tuple[float, Dict[str, float]]:
    """Combine readability and grammar into a clarity score in [0,1]."""
    # textstat might be missing in some environments (e.g., slim deployments)
    try:
        import textstat  # type: ignore
        _has_textstat = True
    except Exception:
        textstat = None  # type: ignore
        _has_textstat = False

    a = answer.strip()
    if not a:
        return 0.0, {"readability": 0.0, "grammar": 0.0}

    # Flesch Reading Ease: roughly [0,100+]; map to [0,1]
    if _has_textstat:
        try:
            fre = textstat.flesch_reading_ease(a)  # type: ignore
            fre = max(0.0, min(100.0, float(fre)))
            readability = fre / 100.0
        except Exception:
            readability = 0.5
    else:
        readability = 0.5

    # Grammar: normalize issue count by length (simple heuristic)
    grammar_score = 1.0
    if lt_tool is not None:
        try:
            matches = lt_tool.check(a)
            num_issues = len(matches)
            length = max(1, len(a.split()))
            # More issues per word => lower score; clamp
            penalty = min(1.0, num_issues / max(10.0, length))
            grammar_score = max(0.0, 1.0 - penalty)
        except Exception:
            grammar_score = 0.6
    else:
        grammar_score = 0.6  # fallback if tool missing

    clarity = 0.6 * readability + 0.4 * grammar_score
    return clarity, {"readability": readability, "grammar": grammar_score}


HEDGING_WORDS = {
    "might", "may", "could", "possibly", "perhaps", "probably", "likely",
    "seems", "appears", "suggests", "approximately", "around", "roughly",
}


def count_hedging(text: str) -> int:
    t = normalize_text(text)
    return sum(len(re.findall(rf"\b{re.escape(w)}\b", t)) for w in HEDGING_WORDS)


# --------------- Fuzzification ---------------
def triangular_low(x: float) -> float:
    # High at 0, decreases to 0 by ~0.6
    if x <= 0.0:
        return 1.0
    if x >= 0.6:
        return 0.0
    return (0.6 - x) / 0.6


def triangular_medium(x: float) -> float:
    # Peak around 0.5, 0 at 0 and 1
    if x <= 0.0 or x >= 1.0:
        return 0.0
    if x == 0.5:
        return 1.0
    if x < 0.5:
        return (x - 0.0) / 0.5
    return (1.0 - x) / 0.5


def triangular_high(x: float) -> float:
    # Low until ~0.4, peak near 1
    if x <= 0.4:
        return 0.0
    if x >= 1.0:
        return 1.0
    return (x - 0.4) / 0.6


def fuzzify_triplet(values: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """For factuality, coherence, clarity -> compute memberships for Low/Med/High, then average.
    Returns (avg_memberships[3], per_metric_memberships[3x3]).
    """
    memberships = []
    for val in values:
        memberships.append(np.array([
            triangular_low(val),
            triangular_medium(val),
            triangular_high(val),
        ], dtype=float))
    per_metric = np.vstack(memberships)  # shape: (3 metrics, 3 levels)
    avg = per_metric.mean(axis=0)
    total = float(np.sum(avg))
    if total > 0:
        avg = avg / total
    return avg, per_metric


def fuzzy_entropy(prob_vector: np.ndarray) -> float:
    """Shannon entropy normalized to [0,1] over 3 categories."""
    eps = 1e-12
    p = np.clip(prob_vector.astype(float), eps, 1.0)
    p = p / p.sum()
    h = -np.sum(p * np.log(p))
    return float(h / math.log(len(p)))


def aggregate_trust(factual: float, coherence: float, clarity: float, entropy: float, hedges: int,
                    w_f: float = 0.4, w_c: float = 0.35, w_cl: float = 0.25,
                    hedge_penalty: float = 0.03, entropy_penalty: float = 0.15) -> float:
    base = w_f * factual + w_c * coherence + w_cl * clarity
    penalty = min(0.5, hedges * hedge_penalty) + entropy * entropy_penalty
    score = max(0.0, min(1.0, base - penalty))
    return score


# --------------- Parser ---------------
def parse_qa_blocks(text: str) -> List[Tuple[str, str]]:
    """Parse multi-line Q/A pairs from textarea.
    Format:
        Q: ...\n
        A: ...\n
    Multiple blocks allowed. Lines between belong to prior Q or A until next marker.
    """
    lines = text.splitlines()
    q_current: List[str] = []
    a_current: List[str] = []
    mode = None  # 'Q' or 'A'
    pairs: List[Tuple[str, str]] = []

    def flush_pair():
        nonlocal q_current, a_current
        q = "\n".join(q_current).strip()
        a = "\n".join(a_current).strip()
        if q and a:
            pairs.append((q, a))
        q_current, a_current = [], []

    for raw in lines:
        line = raw.rstrip("\n")
        if re.match(r"^\s*Q:\s*", line, flags=re.IGNORECASE):
            # Starting a new Q; flush existing if complete
            if q_current or a_current:
                flush_pair()
            content = re.sub(r"^\s*Q:\s*", "", line, flags=re.IGNORECASE)
            q_current.append(content)
            mode = 'Q'
        elif re.match(r"^\s*A:\s*", line, flags=re.IGNORECASE):
            content = re.sub(r"^\s*A:\s*", "", line, flags=re.IGNORECASE)
            a_current.append(content)
            mode = 'A'
        else:
            if mode == 'Q':
                q_current.append(line)
            elif mode == 'A':
                a_current.append(line)
            else:
                # ignore preamble lines
                pass

    # Final flush
    if q_current or a_current:
        flush_pair()

    return pairs


# --------------- Drift detection ---------------
def zscore(value: float, series: List[float]) -> float:
    if not series:
        return 0.0
    arr = np.array(series, dtype=float)
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd == 0:
        return 0.0
    return (value - mu) / sd


# ------------------ UI ------------------
st.set_page_config(page_title="AI Trust Meter", layout="wide")
st.title("AI Trust Meter")
st.caption("Evaluate Q/A factuality, coherence, and clarity – locally, offline.")

embedder, cross_encoder, emb_err, cross_err = _load_models()
lt_tool, lt_err = _load_language_tool()

if emb_err:
    st.warning(f"SentenceTransformer failed to load: {emb_err}")
if cross_err:
    st.info("Optional CrossEncoder unavailable; coherence uses embeddings only.")
if lt_err:
    st.info("language_tool_python unavailable; grammar checks will be approximated.")


# Sidebar controls
with st.sidebar:
    st.subheader("Settings")
    window_size = st.number_input("Entropy window (N)", min_value=5, max_value=200, value=30, step=1)
    z_thresh = st.slider("Drift z-threshold", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
    w_f = st.slider("Weight: Factuality", 0.0, 1.0, 0.4, 0.05)
    w_c = st.slider("Weight: Coherence", 0.0, 1.0, 0.35, 0.05)
    w_cl = st.slider("Weight: Clarity", 0.0, 1.0, 0.25, 0.05)
    hedge_penalty = st.slider("Penalty per hedge", 0.0, 0.1, 0.03, 0.005)
    entropy_penalty = st.slider("Entropy penalty", 0.0, 0.5, 0.15, 0.01)
    # Divider fallback for older Streamlit versions
    try:
        st.divider()
    except Exception:
        st.markdown('---')
    st.markdown("**Local KB Keys**")
    st.write(pd.DataFrame({"key": list(LOCAL_KNOWLEDGE_BASE.keys()), "value": list(LOCAL_KNOWLEDGE_BASE.values())}))


DEFAULT_TEXT = (
    "Q: What is the capital of France?\n"
    "A: The capital of France is Paris.\n\n"
    "Q: Who created Python and when?\n"
    "A: Guido van Rossum created Python around 1991, I think.\n\n"
    "Q: What is the largest planet in our solar system?\n"
    "A: It might be Jupiter, probably.\n"
)

# --- Input area: paste or upload ---
st.subheader("Input")
input_text = st.text_area(
    "Paste Q/A pairs (use 'Q:' and 'A:' markers; multi-line supported)",
    value=DEFAULT_TEXT,
    height=220,
)

uploaded = st.file_uploader("Or upload a .txt or .csv file", type=["txt", "csv"], accept_multiple_files=False)

uploaded_pairs: List[Tuple[str, str]] = []
uploaded_ready = False
if uploaded is not None:
    try:
        if uploaded.type == "text/plain" or uploaded.name.lower().endswith(".txt"):
            content = uploaded.read().decode("utf-8", errors="ignore")
            uploaded_pairs = parse_qa_blocks(content)
            uploaded_ready = True if uploaded_pairs else False
            if not uploaded_ready:
                st.warning("No Q/A pairs detected in uploaded .txt. Ensure 'Q:' and 'A:' markers are used.")
        else:
            # CSV path: expect columns question,answer (case-insensitive). If 2 cols, use first two.
            df_up = pd.read_csv(uploaded)
            cols_lower = [c.lower() for c in df_up.columns]
            q_col = None
            a_col = None
            if "question" in cols_lower and "answer" in cols_lower:
                q_col = df_up.columns[cols_lower.index("question")]
                a_col = df_up.columns[cols_lower.index("answer")]
            elif len(df_up.columns) >= 2:
                q_col, a_col = df_up.columns[0], df_up.columns[1]
            if q_col is None or a_col is None:
                st.error("CSV must have 'question' and 'answer' columns or at least two columns.")
            else:
                uploaded_pairs = [
                    (str(q) if not pd.isna(q) else "", str(a) if not pd.isna(a) else "")
                    for q, a in zip(df_up[q_col].tolist(), df_up[a_col].tolist())
                ]
                # filter empty pairs
                uploaded_pairs = [(q, a) for q, a in uploaded_pairs if q.strip() and a.strip()]
                uploaded_ready = True if uploaded_pairs else False
                if not uploaded_ready:
                    st.warning("No valid Q/A rows found in CSV.")
    except Exception as exc:
        st.error(f"Failed to read uploaded file: {exc}")

col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    run_btn = st.button("Evaluate", use_container_width=True)
with col_btn2:
    clear_btn = st.button("Clear History", use_container_width=True)

run_uploaded_btn = None
if uploaded is not None:
    run_uploaded_btn = st.button("Evaluate Uploaded File", use_container_width=True)


if "history" not in st.session_state:
    st.session_state.history = []  # list of dict rows

if clear_btn:
    st.session_state.history = []


def evaluate_pairs(pairs: List[Tuple[str, str]]):
    for q, a in pairs:
        factual, fact_rationale = simple_fact_check(q, a)
        coherence, coh_rationale = compute_coherence(embedder, cross_encoder, q, a)
        clarity, clarity_breakdown = compute_clarity(a, lt_tool)
        hedges = count_hedging(a)

        avg_memberships, per_metric = fuzzify_triplet((factual, coherence, clarity))
        entropy = fuzzy_entropy(avg_memberships)
        trust = aggregate_trust(
            factual, coherence, clarity, entropy, hedges,
            w_f=w_f, w_c=w_c, w_cl=w_cl,
            hedge_penalty=hedge_penalty, entropy_penalty=entropy_penalty,
        )

        row = {
            "question": q,
            "answer": a,
            "factuality": factual,
            "coherence": coherence,
            "clarity": clarity,
            "clarity_readability": clarity_breakdown.get("readability", np.nan),
            "clarity_grammar": clarity_breakdown.get("grammar", np.nan),
            "hedges": hedges,
            "entropy": entropy,
            "trust": trust,
            "fact_rationale": fact_rationale,
            "coh_rationale": coh_rationale,
            "fuzzy_low": float(avg_memberships[0]),
            "fuzzy_med": float(avg_memberships[1]),
            "fuzzy_high": float(avg_memberships[2]),
        }
        st.session_state.history.append(row)


if run_btn:
    pairs = parse_qa_blocks(input_text)
    if not pairs:
        st.warning("No Q/A pairs detected. Ensure lines start with 'Q:' and 'A:'.")
    else:
        evaluate_pairs(pairs)

if run_uploaded_btn and uploaded_ready:
    evaluate_pairs(uploaded_pairs)


# Convert history to DataFrame
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    # Sliding window drift detection on entropy
    entropies = df["entropy"].tolist()
    window = entropies[-int(window_size):]
    if entropies:
        current_entropy = entropies[-1]
        z = zscore(current_entropy, window)
        if abs(z) >= z_thresh:
            st.warning(f"Entropy drift detected: z={z:.2f} (threshold={z_thresh:.2f})")

    # Charts
    st.subheader("Trends")
    plot_df = df[["trust", "entropy"]].copy()
    plot_df.index = np.arange(1, len(plot_df) + 1)
    st.line_chart(plot_df, height=260)

    # Details table
    st.subheader("Details")
    show_cols = [
        "question", "answer", "trust", "entropy",
        "factuality", "coherence", "clarity",
        "clarity_readability", "clarity_grammar", "hedges",
        "fact_rationale", "coh_rationale",
    ]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

    # Download results
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results as CSV",
        data=csv_bytes,
        file_name="ai_trust_meter_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("Enter Q/A pairs and click Evaluate to see results.")


# ------------------ Footer help ------------------
# How to extend the local knowledge base:
# - Add more key/value pairs to LOCAL_KNOWLEDGE_BASE. Keys should be lowercase
#   and concise (e.g., 'capital of spain' -> 'madrid').
# - Extend `extract_fact_query` with new regex patterns mapping question forms to
#   your KB keys.
#
# To switch to an external fact-check API later:
# - Replace or augment `simple_fact_check` with an API call wrapped in try/except.
# - Keep a local fallback: if the API is unavailable, fall back to LOCAL_KNOWLEDGE_BASE
#   and heuristics so the app remains usable offline.


