import streamlit as st

from haystack import Document
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter

if 'extractive_qa_pipeline' not in st.session_state:
    extractive_qa_pipeline = Pipeline()
    extractive_qa_pipeline.add_component(instance=st.session_state['text_embedder'], name="embedder")
    extractive_qa_pipeline.add_component(instance=st.session_state['retriever'], name="retriever")
    extractive_qa_pipeline.add_component(instance=st.session_state['reader'], name="reader")
    extractive_qa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
    extractive_qa_pipeline.connect("retriever.documents", "reader.documents")
    st.session_state['extractive_qa_pipeline'] = extractive_qa_pipeline

query = st.text_input("Ask your question")

if query:
    results = st.session_state['extractive_qa_pipeline'].run(
    data={"embedder": {"text": query}, "retriever": {"top_k": 3}, "reader": {"query": query, "top_k": 2}}
    )
    st.write(results["reader"]["answers"])