import dotenv
import pandas as pd
from pandas import DataFrame
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader
import os
import requests
from bs4 import BeautifulSoup

load_dotenv()

directory = './store/' + os.environ['DOCUMENT_DIRECTORY']

text = []
source = []

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    extension = os.path.splitext(filename)[1]
    if extension == ".pdf":
        # print(filename)
        reader = PdfReader(filepath)
        pcount = len(reader.pages)
        subtext = ""
        for i in range(0, pcount):
            subtext += reader.pages[i].extract_text()
        text.append(subtext)
    if extension == ".txt":
        with open(f"{directory}/{filename}") as file:
            text.append(file.read())
    source.append(filename)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_text(text)
docs = []
# scrapping_index = []
index = 0
for textItem in text:
    subdocs = text_splitter.split_text(textItem)
    for it in subdocs:
        # docs.append(Document(page_content=it, source=urlList[index]))
        oneDoc = Document(page_content=it, metadata={"source": source[index]})
        docs.append(oneDoc)
        # scrapping_index.append(index)
    index += 1

# with open(f"store/{dirUrl}/urlIndex.txt", "w", encoding="utf-8") as file:
#     for ii in range(0, len(scrapping_index)):
#         file.write("" + str(ii) + ":" + str(scrapping_index[ii]) + "\n")

# for doc_item in docs:
#     document = Document(page_content=doc_item)
#     vector_db.add_documents([document])

# docsDocument = []
# for doc_item in docs:
#     docsDocument.append(Document(page_content=doc_item))

vector_db = None

if os.path.exists(directory + "/index.faiss"):
    vector_db = FAISS.load_local(directory, OpenAIEmbeddings())
else:   
    vector_db = FAISS.from_documents(docs, OpenAIEmbeddings())

vector_db.save_local(directory)
print("completed")