import streamlit as st

st.title("🧭 Exploration de thèmes avec BERTopic")

topic_input = st.slider("Choisir un topic ID", min_value=0, max_value=len(set(topics))-1)
docs = topic_model.get_representative_docs(topic_input)

st.write(f"💬 Exemples de documents du topic {topic_input}")
for doc in docs:
    st.text(doc)