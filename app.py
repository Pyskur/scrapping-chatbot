import os
from flask import Flask, request, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import pickle

load_dotenv()

url = os.environ['SCRAP_WEBSITE']

app = Flask(__name__)
CORS(app)

dir_name = "./store/"
if "https" in url:
    dir_name += url.replace("https", "").replace("/", "").replace(":", "")
else:
    dir_name += url.replace("http", "").replace("/", "").replace(":", "")

prompt_template = """Use the following pieces of context to answer the question at the end. This context is for selling, so answer any questions if a customer ask for selling, answer in details about it. If selling is not available, answer to customer in most similar thing which is available for selling. If question is not related to context, just say that it is not related to context, don't try to make up an answer.

{context}

Question: {question}
Answer in English:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt=PROMPT)
chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")
if os.path.exists(dir_name + "/index.faiss"):
    docsearch = FAISS.load_local(dir_name, OpenAIEmbeddings())
else:
    docsearch = FAISS.from_documents([Document(page_content="I don\'t know\n\n")], OpenAIEmbeddings())

@app.route('/api/chat', methods=['POST'])
def chat():
    query = request.form["prompt"]
    docs = docsearch.similarity_search(query)
    # completion = chain.run(input_documents=docs, question=query)
    completion = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    print(completion)
    return {"answer": "123" }

if __name__ == '__main__':
    app.run(debug=True)