from dotenv import load_dotenv
import os 
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
#for large pages we convert it into chunks and store into vector databases

data = PyPDFLoader(r"C:\Users\Shivansh\Desktop\schola\deep-learning-book-2025.pdf")
documents = data.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)
chunks = splitter.split_documents(documents)

prompt = ChatPromptTemplate.from_messages([
    ("system" , """you are an ai that summarizeds the text
    
    """),
    (
        "user" , """{data} 
        """
    )

])
template = prompt.format_messages(data = chunks.page_content)

model = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider='groq',
    api_key= os.getenv("GROQ_API_KEY")
)
response = model.invoke(template)
print(response.content)
