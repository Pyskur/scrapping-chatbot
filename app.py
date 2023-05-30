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
from langchain.chains.conversation.memory import ConversationBufferMemory
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

prompt_template = """Your name is Ansi and you work for {url}. You are going to behave as the best sales rep in the whole universe. You will be integrated in a chat so customers will be talking to you via a chat window. Your goal is to keep the conversation going with the customer unless the customer agrees to buy something. You will keep re igniting the conversation and ask questions from the customer unless he/she is convinced to buy something from the context or he/she gets the answer they were looking for. You have all the data in the context. Use the following pieces of context to serve the customers. This context is for selling and assisting in customers. If an item/service is not available for sale then recommend the most similar thing to the customer that is available with in the context. Example: if a customer asks for an android phone and android phone is not available then recomend another phone. Or if a customer asks for web development services and that is not available but mobile development services is available so offer that instead. If question is not related to context, just say that it is not related to context in a polite way, don't try to make up an answer. In no circumstances you will deviate from the context and try to make up answers outside the context. Create a final answer with references ("SOURCES").

=========
{summaries}
=========
{chat_history}
Human: {question}
Chatbot: 
Answer in language of human's last question:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["chat_history", "summaries", "question", "url"]
)

llm = OpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
chain = load_qa_with_sources_chain(llm, chain_type="stuff", verbose=True, prompt=PROMPT, memory=memory)
if os.path.exists(dir_name + "/index.faiss"):
    docsearch = FAISS.load_local(dir_name, embeddings)
else:
    docsearch = FAISS.from_documents([Document(page_content="I don\'t know\n\n")], embeddings)

@app.route('/api/chat', methods=['POST'])
def chat():
    query = request.form["prompt"]
    docs = docsearch.similarity_search(query)
    completion = chain({"input_documents": docs, "question": query, "url": url}, return_only_outputs=True)
    return {"answer": completion["output_text"] }

if __name__ == '__main__':
    app.run(debug=True)