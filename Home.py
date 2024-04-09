import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
from datasets import load_dataset
import pandas as pd
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Pipeline
from haystack.components.writers import DocumentWriter

if 'data' not in st.session_state:
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    st.session_state['data'] = pd.DataFrame(dataset)

if 'openai_key' not in st.session_state:
    st.session_state["openai_key"] = os.getenv("OPENAI_API_KEY")

if 'small_model' not in st.session_state:
    small_model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    document_embedder = SentenceTransformersDocumentEmbedder(model=small_model)
    st.session_state['document_embedder'] = document_embedder
    text_embedder = SentenceTransformersTextEmbedder(model=small_model)
    st.session_state['text_embedder'] = text_embedder
    st.session_state['small_model'] = small_model

if 'documents' not in st.session_state:
    documents = documents = [Document(content=st.session_state['data']["content"][index], meta=st.session_state['data']["meta"][index]) for index in st.session_state['data'].index]
    st.session_state['documents'] = documents

if 'document_store' not in st.session_state:
    document_store = InMemoryDocumentStore()
    st.session_state['document_embedder'].warm_up()
    # st.session_state['document_store'] = document_store
    # docs_with_embeddings = st.session_state['document_embedder'].run(st.session_state['documents'])
    # document_store.write_documents(docs_with_embeddings["documents"])
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(instance=st.session_state['document_embedder'], name="embedder")
    indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
    indexing_pipeline.connect("embedder.documents", "writer.documents")
    indexing_pipeline.run({"documents": st.session_state['documents']})
    st.session_state["indexing_pipeline"] = indexing_pipeline
    st.session_state['document_store'] = document_store
    

if 'retriever' not in st.session_state:
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    st.session_state['retriever'] = retriever

if 'reader' not in st.session_state:
    reader = ExtractiveReader()
    reader.warm_up()
    st.session_state['reader'] = reader

st.title("RAG Demo")

st.image("B_ROBOTS2019.webp")
