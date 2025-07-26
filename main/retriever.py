from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field

from store import PERSIST_DIRECTORY
from langchain_core.tools import tool


import config
@tool
def retrieve(query):
    """ Retrieve the data related to question asked
         Args:
        query: Search string query
     """
    print("---- retrieve:" + query)
    embeddings = OllamaEmbeddings(model=config.OLLAMA_MODEL)
    chroma = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = chroma.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )
    result = retriever.invoke(query)
    return result

# retrieve("What is madurai?")
