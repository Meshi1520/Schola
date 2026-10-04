#we will load documents here 
# the split documents are the text files that are nodes
# even from some websites as well 
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(

    chunk_size = 10,
    chunk_overlap = 1
)



from langchain_community.document_loaders import TextLoader
data = TextLoader(r"C:\Users\Shivansh\Desktop\schola\document loaders\notes.txt")
#a document has metadata and page content 
#for text data therer is only one document
document = data.load()
chunks = splitter.split_documents(document)
for i in chunks:
    print(i.page_content)


