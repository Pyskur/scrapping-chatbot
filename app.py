import os
from flask import Flask, request, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType

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

prompt_template = """You're a sales chatbot from {url}. Use the following pieces of context to answer the question at the end. This context is for selling, so answer any questions if a customer ask for selling, answer in details about it. If selling is not available, answer to customer in most similar thing which is available for selling. If question is not related to context, just say that it is not related to context, don't try to make up an answer.

{context}

Question: {question}
Answer in English:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question", "url"]
)

llm = OpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

if os.path.exists(dir_name + "/index.faiss"):
    docsearch = FAISS.load_local(dir_name, embeddings)
else:
    docsearch = FAISS.from_documents([Document(page_content="I don\'t know\n\n")], embeddings)

def get_result(query):
    docs = docsearch.similarity_search(query)
    completion = chain.run(input_documents=docs, question=query, url=url)
    return completion

tools = [
    Tool(
        name = "Current Search",
        func=get_result,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]
memory = ConversationBufferMemory(memory_key="chat_history")
agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

@app.route('/api/chat', methods=['POST'])
def chat():
    query = request.form["prompt"]
    # docs = docsearch.similarity_search(query)
    # completion = chain.run(input_documents=docs, question=query, url=url)
    completion = agent_chain.run(query)
    return {"answer": completion }

if __name__ == '__main__':
    app.run(debug=True)