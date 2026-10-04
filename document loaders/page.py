#we page summarization web page loader
from langchain_community.document_loaders import WebBaseLoader
url = "https://www.bing.com/search?pglt=299&q=amazon+shopping&cvid=58ec233da6504309a49a44ad09fcd4a0&gs_lcrp=EgRlZGdlKgcIABAAGPkHMgcIABAAGPkHMgYIARAuGEAyBggCEAAYQDIGCAMQABhAMgYIBBAAGEAyBggFEAAYQDIGCAYQLhhAMgYIBxBFGEEyBggIEAUYQNIBCDE3MzlqMGo3qAIIsAIB&FORM=ANNTA1&PC=ACTS"
data =  WebBaseLoader(url)
documents = data.load()
print(documents)
