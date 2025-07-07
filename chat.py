from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Load FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore", embedding_model, allow_dangerous_deserialization=True)

# Load Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192"
)

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a medical chatbot. Use only the following context to answer the user's question.

Context: {context}
Question: {question}

Helpful Answer:"""
)

# Create chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt_template}
)

# CLI
print("🤖 Medical PDF Chatbot (type 'exit' to quit)")
while True:
    query = input("📥 Question: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = qa_chain.run(query)
    print(f"📤 Answer: {response}\n")
