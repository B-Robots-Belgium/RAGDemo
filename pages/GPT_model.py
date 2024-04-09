import streamlit as st

import os
from getpass import getpass
from haystack.components.generators import OpenAIGenerator
from haystack import Document
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter

from haystack.components.builders import PromptBuilder

template = """
Using only the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

prompt_builder = PromptBuilder(template=template)

generator = OpenAIGenerator(model="gpt-3.5-turbo")

if "basic_rag_pipeline" not in st.session_state:
    retriever = InMemoryEmbeddingRetriever(st.session_state['document_store'])
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    basic_rag_pipeline = Pipeline()
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", generator)
    basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
    basic_rag_pipeline.connect("prompt_builder", "llm")
    st.session_state["basic_rag_pipeline"] = basic_rag_pipeline

query = st.text_input("Ask your question")

if query:
    results = st.session_state["basic_rag_pipeline"].run(
    data={"text_embedder": {"text": query}, "retriever": {"top_k": 3}, "prompt_builder": {"question": query}})
    st.write(results["llm"]["replies"])
    st.write(results)