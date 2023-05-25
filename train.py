import dotenv
import pandas as pd
from pandas import DataFrame
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
import os
import requests
from bs4 import BeautifulSoup

load_dotenv()

urlList = []
url = os.environ['SCRAP_WEBSITE']

urlList.append(url)

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

urlOtherFormat = ""

if "https" in url:
    urlOtherFormat = url.replace("https", "http")
else:
    urlOtherFormat = url.replace("http", "https")

links = soup.find_all('a')
for link in links:
    targetLink = link.get('href')
    if url in targetLink or urlOtherFormat in targetLink:
        urlList.append(targetLink)

urlList = list(set(urlList))
print(urlList)

document = []

for item in urlList:
    print("scrapping : " + item + "\n")
    loader = WebBaseLoader(item)
    data = loader.load()
    for subdata in data:
        document.append(subdata)

text = ""

for doc in document:
    text += doc.page_content

with open("scrapping.txt", "w", encoding="utf-8") as file:
    # Write the string to the file
    file.write(text)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_text(text)
# for doc_item in docs:
#     document = Document(page_content=doc_item)
#     vector_db.add_documents([document])

docsDocument = []
for doc_item in docs:
    docsDocument.append(Document(page_content=doc_item))

dir_name = "./store/"
if "https" in url:
    dir_name += url.replace("https", "").replace("/", "").replace(":", "")
else:
    dir_name += url.replace("http", "").replace("/", "").replace(":", "")

vector_db = None

if os.path.exists(dir_name + "/index.faiss"):
    vector_db = FAISS.load_local(dir_name, OpenAIEmbeddings())
else:
    vector_db = FAISS.from_documents(docsDocument, OpenAIEmbeddings())

vector_db.save_local(dir_name)
print("completed")