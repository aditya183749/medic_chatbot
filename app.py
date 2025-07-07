from flask import Flask, request, render_template
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# Load env vars
load_dotenv()

app = Flask(__name__)

# Load embedding and vector DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore", embedding_model, allow_dangerous_deserialization=True)

# Setup Groq LLM
llm = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash"
)

# Prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a medical chatbot. Use only the following context to answer the user's question.

Context: {context}
Question: {question}

Helpful Answer:"""
)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt_template}
)

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        query = request.form["query"]
        response = qa_chain.run(query)
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(port=10000, host="0.0.0.0")

