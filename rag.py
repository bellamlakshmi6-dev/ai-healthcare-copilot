import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

print("🚀 Starting AI...")

# Load API key
load_dotenv()

# Load text data
loader = TextLoader("sample.txt")
documents = loader.load()
print("✅ Text loaded")

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)
print("✅ Text split")

# Create embeddings
embeddings = OpenAIEmbeddings()
print("✅ Embeddings ready")

# Create vector DB
db = FAISS.from_documents(docs, embeddings)
print("✅ Vector DB ready")

# Create retriever (IMPORTANT: fetch more data)
retriever = db.as_retriever(search_kwargs={"k": 10})

# Load LLM
llm = ChatOpenAI(model="gpt-4o-mini")
print("✅ LLM ready")

print("🔥 Reached chatbot loop")

# Chat loop
while True:
    q = input("Ask your question: ")

    # Retrieve relevant data
    docs = retriever.invoke(q)
    context = "\n".join([doc.page_content for doc in docs])

    # Strong prompt for accuracy
    prompt = f"""
    You are a healthcare data assistant.

    Use ONLY the provided dataset.
    If asked to count, COUNT carefully.
    Do NOT guess.
    If answer not found, say "Not found".

    DATA:
    {context}

    QUESTION:
    {q}
    """

    response = llm.invoke(prompt)

    print("🤖 Answer:", response.content)