#there are many embedding moedls 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
docs = [
    Document(page_content= "hello nice to meet you sir", metadata = {"source" : "sir"}),
    Document(page_content= "hello nice to meet you bhaiya",metadata = {"source" : "bhaiya"}),
    Document(page_content= "hello nice to meet you papa ji",metadata = {"source" : "papa"})

]

embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
# now we make a vector store
vectorstore = Chroma.from_documents(
    documents= docs,
    embedding= embedding_model,
    persist_directory= "chroma-db"

)


#we create a directiry to save by using our system's space 
#we store a lot of data ;ike page content and meta data so we are using chroma sqllite
result = vectorstore.similarity_search("papa ji ",k = 2)
for r in result:
    print(r)

retriever = vectorstore.as_retriever()

docs = retriever.invoke("tell me about papa ji");
for d in docs:
    print(d.page_content)
