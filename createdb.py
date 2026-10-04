#data ingestion
#load the pdf
#split it into chunks
#create embeddings
#store it into db
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()
data  = PyPDFLoader(r"C:\Users\Shivansh\Desktop\schola\deep-learning-book-2025.pdf")
docs = data.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10000,
    chunk_overlap = 1
)
chunks = splitter.split_documents(docs)
embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma.from_documents(
    documents = docs,
    embedding_model = embedding_model,
    persist_directory= "chroma_db"
)
