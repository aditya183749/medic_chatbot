🩺 Medic Chatbot

A web-based AI medical chatbot that answers user questions using context extracted from a medical encyclopedia PDF. Built using Flask, LangChain, FAISS, and Google Gemini.


## 🚀 Features

- Conversational medical Q&A interface  
- Context-aware responses using vector similarity search (FAISS)  
- Embedding generation via HuggingFace Transformers  
- Large Language Model (LLM) responses powered by Google Gemini  
- Simple and clean web-based UI  


## 🧠 How It Works

1. A medical encyclopedia PDF is parsed and split into chunks.  
2. Each chunk is embedded using HuggingFace's `sentence-transformers`.  
3. FAISS creates a vector store to enable fast semantic search.  
4. User questions are matched to relevant chunks.  
5. The selected context is passed to **Gemini** via LangChain for answering.


🛠️ Setup

 1. Clone the repository

```bash
git clone https://github.com/yourusername/medic_chatbot.git
cd medic_chatbot

2. Install dependencies
pip install -r requirements.txt

3. Set up environment variables
Create a .env file and add your Google Gemini API key:
env
GEMINI_API_KEY=your_google_api_key_here

4. Prepare the vector store
python create_memory_from_pdf.py

5. Run the app
python app.py
The app will be live at: http://localhost:10000

📁 File Structure
medic_chatbot/
├── app.py                        # Flask app and core logic
├── chat.py                       # (Optional) CLI chatbot
├── create_memory_from_pdf.py     # Embeds and indexes the PDF
├── data/                         # PDF files
├── vectorstore/                  # Saved FAISS index
├── templates/
│   └── index.html                # Web interface template
├── requirements.txt              # Python dependencies
└── .env                          # API key (not committed)
📋 Requirements
Python 3.8+

All dependencies listed in requirements.txt

⚠️ Disclaimer
This chatbot is for informational purposes only and does not provide real medical advice.
Always consult a licensed healthcare provider for any medical condition.


