from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import config

PERSIST_DIRECTORY = "./vector-store"

def load_from_wikipedia(query):
    docs = WikipediaLoader(query=query, load_max_docs=2).load()
    return docs


def index(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return text_splitter.split_documents(docs)


def store(topic):
    docs = load_from_wikipedia(topic)
    embeddings = OllamaEmbeddings(model=config.OLLAMA_MODEL)
    vector_store = Chroma.from_documents(documents=docs,
                                         embedding=embeddings,
                                         persist_directory=PERSIST_DIRECTORY)
    ids = vector_store.add_documents(documents=index(docs))


store(config.TOPIC)