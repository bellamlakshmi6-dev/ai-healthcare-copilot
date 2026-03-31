import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import tempfile
import speech_recognition as sr

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="Healthcare AI", layout="wide")

# 🎨 UI Styling
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f172a, #1e293b); color: white; }
.title { text-align: center; font-size: 40px; color: #38bdf8; }
.user-msg { background:#38bdf8; color:black; padding:10px; border-radius:10px; margin:5px; text-align:right;}
.bot-msg { background:#1e293b; padding:10px; border-radius:10px; margin:5px; border-left:4px solid #38bdf8;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 Healthcare AI Copilot</div>', unsafe_allow_html=True)

# 📂 Upload dataset
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Dataset Preview")
    st.dataframe(df.head())

    # Convert to text
    text_data = ""
    for _, row in df.iterrows():
        text_data += " ".join([str(val) for val in row]) + "\n"

    with open("temp.txt", "w") as f:
        f.write(text_data)

    data_file = "temp.txt"
else:
    data_file = "sample.txt"

# 🧠 Load system
@st.cache_resource
def load_system(file_path):
    loader = TextLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    retriever = db.as_retriever(search_kwargs={"k": 10})
    llm = ChatOpenAI(model="gpt-4o-mini")

    return retriever, llm

retriever, llm = load_system(data_file)

# 🎤 Voice input
st.write("### 🎤 Voice Input")
if st.button("Record Question"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            st.success(f"You said: {query}")
        except:
            st.error("Could not understand audio")
            query = ""
else:
    query = st.text_input("💬 Ask your question:")

# 💬 Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if query:
    st.session_state.messages.append(("user", query))

    # ⏳ Typing animation
    with st.spinner("🤖 AI is thinking..."):
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer ONLY using dataset.
        Do NOT guess.

        DATA:
        {context}

        QUESTION:
        {query}
        """

        response = llm.invoke(prompt).content

    st.session_state.messages.append(("bot", response))

# 💬 Display chat
for role, msg in st.session_state.messages:
    if role == "user":
        st.markdown(f'<div class="user-msg">🧑 {msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 {msg}</div>', unsafe_allow_html=True)

# 📊 Charts Dashboard
st.write("## 📊 Data Insights")

if uploaded_file:
    st.bar_chart(df.select_dtypes(include='number'))
    st.line_chart(df.select_dtypes(include='number'))