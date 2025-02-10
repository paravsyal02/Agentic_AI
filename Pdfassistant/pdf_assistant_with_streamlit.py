import streamlit as st
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2 
from phi.embedder.google import GeminiEmbedder
from phi.llm.groq import Groq 
import os
from dotenv import load_dotenv

#load env var
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

#Setting Knowledge Base
knowledge_base = PDFUrlKnowledgeBase(
    urls = ["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db = PgVector2(
        collection = "recipes",
        db_url = db_url,
        embedder = GeminiEmbedder(),
    ) 
)
knowledge_base.load()

#Set up Storage
storage = PgAssistantStorage(table_name="pdf_assistant", db_url=db_url)

#Initialize agent
assistant = Assistant(
    run_id=None,
    user_id="user",
    knowledge_base=knowledge_base, 
    storage=storage,
    show_tool_calls=True,
    search_knowledge=True,
    read_chat_history=True,
    llm=Groq(model="llama-3.3-70b-versatile", name="Groq", embedder=GeminiEmbedder()),
)

#Streamlit UI
st.title("PDF AI Assistant")
st.write("Ask questions about the PDF content!")

#User Input
user_input = st.text_input("Enter your question:")
if st.button("Ask"):
    if user_input:
        response = assistant.run(user_input)  # Get generator response
        response_text = "".join(response)  # Convert generator to string
        st.write("**Response:**", response_text)  # Display response

    else:
        st.warning("Please enter a question!")
