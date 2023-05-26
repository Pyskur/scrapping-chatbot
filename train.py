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
dirUrl = url.replace("https", "").replace("/", "").replace(":", "")

if os.path.exists(f"store/{dirUrl}/urlList.txt"):
    with open(f"store/{dirUrl}/urlList.txt", "r", encoding="utf-8") as file:
        for line in file:
            urlList.append(line.replace("\n", ""))
else:

    urlList.append(url)

    def scrap_page(curlink):
        response = requests.get(curlink)
        soup = BeautifulSoup(response.text, 'html.parser')

        links = soup.find_all('a')
        if len(links) == 0:
            return 
        for link in links:
            targetLink = link.get('href')
            if targetLink and url in targetLink and '.' not in targetLink.split('/')[-1] and targetLink not in urlList:
                urlList.append(targetLink)
                print(targetLink + "\n")
                scrap_page(targetLink)

    scrap_page(url)
    urlList = list(set(urlList))

    with open(f"store/{dirUrl}/urlList.txt", "w", encoding="utf-8") as file:
        for urlItem in urlList:
            file.write(urlItem + "\n")
         
print(urlList)

document = []
text = []

for item in urlList:
    print("scrapping : " + item + "\n")
    loader = WebBaseLoader(item)
    data = loader.load()
    subtext = ""
    for subdata in data:
        subtext += subdata.page_content
    text.append(subtext)

# for doc in document:
#     text += doc.page_content

# with open("scrapping.txt", "w", encoding="utf-8") as file:
#     # Write the string to the file
#     file.write(text)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_text(text)
docs = []
scrapping_index = []
index = 0
for textItem in text:
    subdocs = text_splitter.split_text(textItem)
    for it in subdocs:
        docs.append(it)
        scrapping_index.append(index)
    index += 1

with open(f"store/{dirUrl}/urlIndex.txt", "w", encoding="utf-8") as file:
    for ii in range(0, len(scrapping_index)):
        file.write("" + str(ii) + ":" + str(scrapping_index[ii]) + "\n")

# for doc_item in docs:
#     document = Document(page_content=doc_item)
#     vector_db.add_documents([document])

docsDocument = []
for doc_item in docs:
    docsDocument.append(Document(page_content=doc_item))

vector_db = None
dir_name = "./store/" + dirUrl

if os.path.exists(dir_name + "/index.faiss"):
    vector_db = FAISS.load_local(dir_name, OpenAIEmbeddings())
else:   
    vector_db = FAISS.from_documents(docsDocument, OpenAIEmbeddings())

vector_db.save_local(dir_name)
print("completed")