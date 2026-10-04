# from langchain_community.document_loaders import PyPDFLoader
# data = PyPDFLoader(r"C:\Users\Shivansh\Desktop\schola\document loaders\GRU.pdf")
# documents = data.load()
# print(len(documents))
from dotenv import load_dotenv
import os 
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import TokenTextSplitter

data = PyPDFLoader(r"C:\Users\Shivansh\Desktop\schola\document loaders\GRU.pdf")
documents = data.load()
splitter = TokenTextSplitter(
    chunk_size = 10,
    chunk_overlap = 1,

)
chunks = splitter.split_documents(documents)
print(len(chunks))
print(chunks[0])

# prompt = ChatPromptTemplate.from_messages([
#     ("system" , """you are an ai that summarizeds the text
    
#     """),
#     (
#         "user" , """{data} 
#         """
#     )

# ])
# template = prompt.format_messages(data = documents[0].page_content)

# model = init_chat_model(
#     model="openai/gpt-oss-120b",
#     model_provider='groq',
#     api_key= os.getenv("GROQ_API_KEY")
# )
# response = model.invoke(template)
# print(response.content)
