import streamlit as st
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd

# --- Données d'exemple ---
corpus = [
    "Le président a tenu une conférence sur la sécurité nationale.",
    "La technologie des batteries lithium-ion avance rapidement.",
    "Le tournoi de football européen démarre ce week-end.",
    "Les élections auront lieu en juin avec des enjeux majeurs.",
    "Les avancées en intelligence artificielle transforment l'industrie."
]

# --- Nettoyage (léger pour l'exemple) ---
corpus = [doc.strip() for doc in corpus if doc.strip()]

# --- Modèle d'embeddings multilingue ---
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# --- Initialisation du modèle BERTopic ---
topic_model = BERTopic(embedding_model=embedding_model, language="french")
topics, probs = topic_model.fit_transform(corpus)

# --- Interface Streamlit ---
st.title("🧭 Exploration thématique avec BERTopic")

# Filtrage des topics valides (≠ -1)
topics_valides = sorted([t for t in set(topics) if t != -1])

if topics_valides:
    topic_id = st.selectbox("Sélectionnez un topic à explorer :", topics_valides)

    # Récupération des documents appartenant au topic
    doc_info = topic_model.get_document_info(corpus)
    docs_topic = doc_info[doc_info.Topic == topic_id]["Document"].tolist()

    st.subheader(f"📌 Documents du topic {topic_id}")
    for doc in docs_topic:
        st.write(f"• {doc}")

    st.subheader("🔍 Visualisation du topic")
    st.plotly_chart(topic_model.visualize_barchart(top_n_topics=len(topics_valides)))
else:
    st.warning("Aucun topic n'a été détecté. Veuillez augmenter la taille du corpus.")
