import streamlit as st
import pickle
import re
import os
import html
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Emotion Detector | by Jaswant",
    page_icon="◐",
    layout="centered",
)

EMOTION_ORDER = ["sadness", "anger", "love", "surprise", "fear", "joy"]

PALETTE = {
    "sadness":  {"color": "#3D5A80", "glyph": "≈"},
    "anger":    {"color": "#BC4B2C", "glyph": "▲"},
    "love":     {"color": "#A8305F", "glyph": "♥"},
    "surprise": {"color": "#D9A404", "glyph": "!"},
    "fear":     {"color": "#5B4B8A", "glyph": "✷"},
    "joy":      {"color": "#2F9C5A", "glyph": "○"},
}

INK = "#20242B"
PAPER = "#EEF1EF"
CARD = "#FFFFFF"
MUTED = "#6B7178"


st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Work+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{
            font-family: 'Work Sans', sans-serif;
            color: {INK};
        }}
        .stApp {{
            background: {PAPER};
        }}
        .ed-eyebrow {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {MUTED};
            margin-bottom: 0.2rem;
        }}
        .ed-title {{
            font-family: 'Fraunces', serif;
            font-weight: 700;
            font-size: 2.6rem;
            line-height: 1.05;
            margin-bottom: 0.1rem;
            color: {INK};
        }}
        .ed-byline {{
            font-family: 'Work Sans', sans-serif;
            font-size: 0.92rem;
            color: {MUTED};
            margin-bottom: 1.1rem;
        }}
        .ed-sub {{
            font-size: 1rem;
            color: {INK};
            opacity: 0.82;
            margin-bottom: 1.6rem;
            max-width: 46em;
        }}
        div[data-testid="stTextArea"] textarea {{
            font-family: 'Work Sans', sans-serif;
            font-size: 1rem;
            border-radius: 4px;
            border: 1.5px solid #D7DAD8;
            background: {CARD};
            color: {INK} !important;
            -webkit-text-fill-color: {INK} !important;
            caret-color: {INK};
        }}
        div[data-testid="stTextArea"] textarea::placeholder {{
            color: {MUTED} !important;
            opacity: 1;
        }}
        div[data-testid="stTextArea"] textarea:focus {{
            border-color: {INK};
            box-shadow: none;
        }}
        div.stButton > button {{
            font-family: 'Work Sans', sans-serif;
            font-weight: 600;
            border-radius: 3px;
            border: 1.5px solid {INK};
            transition: none;
        }}
        div.stButton > button[kind="primary"] {{
            background: {INK};
            color: {PAPER};
        }}
        div.stButton > button[kind="secondary"] {{
            background: transparent;
            color: {INK};
        }}
        .ed-result-card {{
            background: {CARD};
            border: 1px solid #DEE1DF;
            border-left: 6px solid var(--accent);
            border-radius: 4px;
            padding: 1.4rem 1.6rem;
            margin-top: 1.2rem;
        }}
        .ed-result-label {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {MUTED};
        }}
        .ed-result-emotion {{
            font-family: 'Fraunces', serif;
            font-weight: 700;
            font-size: 2.1rem;
            color: var(--accent);
            margin: 0.1rem 0 0.2rem 0;
        }}
        .ed-result-conf {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
            color: {MUTED};
        }}
        .ed-spectrum-wrap {{
            margin-top: 1.3rem;
        }}
        .ed-spectrum-title {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {MUTED};
            margin-bottom: 0.5rem;
        }}
        .ed-spectrum {{
            display: flex;
            gap: 4px;
            height: 54px;
        }}
        .ed-seg {{
            flex: 1;
            border-radius: 3px;
            position: relative;
            display: flex;
            align-items: flex-end;
            justify-content: center;
            padding-bottom: 4px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem;
            color: white;
            overflow: hidden;
        }}
        .ed-seg-marker {{
            position: absolute;
            top: -22px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 1.1rem;
        }}
        .ed-spectrum-labels {{
            display: flex;
            gap: 4px;
            margin-top: 0.35rem;
        }}
        .ed-spectrum-labels span {{
            flex: 1;
            text-align: center;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.62rem;
            color: {MUTED};
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        .ed-footer {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            color: {MUTED};
            margin-top: 2.2rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    with open(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(BASE_DIR, "logistic_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "emotion_numbers.pkl"), "rb") as f:
        emotion_map = pickle.load(f)
    reverse_map = {v: k for k, v in emotion_map.items()}
    return vectorizer, model, reverse_map


vectorizer, model, reverse_map = load_artifacts()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def render_spectrum(probs_by_emotion: dict, predicted: str) -> str:
    segs = []
    labels = []
    for emo in EMOTION_ORDER:
        p = probs_by_emotion.get(emo, 0.0)
        color = PALETTE[emo]["color"]
        glyph = PALETTE[emo]["glyph"]
        opacity = 0.22 + 0.78 * p
        marker = f'<div class="ed-seg-marker">{glyph}</div>' if emo == predicted else ""
        segs.append(
            f'<div class="ed-seg" style="background:{color}; opacity:{opacity:.2f};">'
            f'{marker}{p*100:.0f}%</div>'
        )
        labels.append(f"<span>{emo}</span>")
    return (
        '<div class="ed-spectrum">' + "".join(segs) + "</div>"
        '<div class="ed-spectrum-labels">' + "".join(labels) + "</div>"
    )


st.markdown('<div class="ed-eyebrow">Text → Signal → Emotion</div>', unsafe_allow_html=True)
st.markdown('<div class="ed-title">Emotion Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="ed-byline">Built by Jaswant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ed-sub">Write a sentence and the model reads it against six '
    'emotional registers — sadness, anger, love, surprise, fear, joy — '
    'and shows where it lands.</div>',
    unsafe_allow_html=True,
)

text_input = st.text_area(
    "Enter your text",
    height=110,
    placeholder="e.g. I can't believe how amazing today turned out to be!",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 1])
with col1:
    predict_clicked = st.button("Detect emotion", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", type="secondary", use_container_width=True)

if clear_clicked:
    st.rerun()

if predict_clicked:
    if not text_input.strip():
        st.warning("Type something first — the detector needs a sentence to read.")
    else:
        cleaned = clean_text(text_input)
        X = vectorizer.transform([cleaned])

        pred_label = model.predict(X)[0]
        pred_emotion = reverse_map[pred_label]
        accent = PALETTE[pred_emotion]["color"]

        probs_by_emotion = {}
        top_conf = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            probs_by_emotion = {reverse_map[i]: probs[i] for i in range(len(probs))}
            top_conf = probs_by_emotion[pred_emotion]

        conf_text = f"{top_conf*100:.0f}% confidence" if top_conf is not None else ""

        st.markdown(
            f"""
            <div class="ed-result-card" style="--accent: {accent};">
                <div class="ed-result-label">Reading</div>
                <div class="ed-result-emotion">{html.escape(pred_emotion.capitalize())}</div>
                <div class="ed-result-conf">{conf_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if probs_by_emotion:
            st.markdown(
                '<div class="ed-spectrum-wrap">'
                '<div class="ed-spectrum-title">Full spectrum</div>'
                + render_spectrum(probs_by_emotion, pred_emotion)
                + "</div>",
                unsafe_allow_html=True,
            )

            with st.expander("Exact probabilities"):
                prob_df = pd.DataFrame(
                    {
                        "Emotion": [e.capitalize() for e in EMOTION_ORDER],
                        "Probability": [probs_by_emotion[e] for e in EMOTION_ORDER],
                    }
                ).sort_values("Probability", ascending=False).reset_index(drop=True)
                st.dataframe(
                    prob_df.style.format({"Probability": "{:.2%}"}),
                    use_container_width=True,
                    hide_index=True,
                )

st.markdown(
    '<div class="ed-footer">Emotion Detector — TF-IDF + Logistic Regression — made by Jaswant</div>',
    unsafe_allow_html=True,
)