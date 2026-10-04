from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
docs = [
     Document(page_content= "hello nice to meet you sir", metadata = {"source" : "sir"}),
        Document(page_content= "hello nice to meet you bhaiya",metadata = {"source" : "bhaiya"}),
        Document(page_content= "hello nice to meet you papa ji",metadata = {"source" : "papa"})
    
]
embeddings = HuggingFaceBgeEmbeddings()

vectorstore = Chroma.from_documents(docs,embeddings)

mmr_retriever = vectorstore.as_retriever(
    search_type = 'mmr',
    search_kwargs = {"k" : 3}

)