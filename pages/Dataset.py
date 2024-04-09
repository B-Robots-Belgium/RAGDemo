import streamlit as st
import pandas as pd
from io import StringIO
import uuid
from haystack import Document

st.title("Data Analysis")
new_files = None

def Add_Item():
    documents = []
    for file in new_files:
        stringio = StringIO(file.getvalue().decode("utf-8"))
        file_text = stringio.read()
        split_text = [file_text[i:i+1000] for i in range(0, len(file_text), 1000)]
        for split in split_text:
            st.session_state["data"].loc[len(st.session_state["data"])] = {
                "id": uuid.uuid4(),
                "content": split,
                "content_type": "text",
                "meta": {
                    "source": "Uploaded via interface"
                },
                "id_hash_keys": None,
                "score": None,
                "embedding": None
            }
            documents.append(Document(content=split, meta={"source": "Uploaded via interface"}))
    st.session_state['indexing_pipeline'].run({"documents": documents})

st.data_editor(data=st.session_state['data'], num_rows="dynamic",
               column_config= {
                   "id": "ID",
                   "content": "Text content",
                   "meta": "Sources"
               }, width=850, height=850)

new_files = st.file_uploader("Add new items to the dataset", accept_multiple_files=True)

if new_files:
    st.button(label="Update dataset", on_click=Add_Item)