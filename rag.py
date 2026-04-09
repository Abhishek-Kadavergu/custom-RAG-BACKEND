# rag.py

from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class RAGPipeline:
    def __init__(self, api_key: str):
        # 1. LLM
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=api_key
        )

        # 2. Load data
        loader = TextLoader("data/profile.txt", encoding="utf-8")
        docs = loader.load()

        # 3. Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )
        split_docs = splitter.split_documents(docs)

        # 4. Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
        )

        # 5. Vector DB
        self.vector_store = Chroma.from_documents(
            split_docs,
            embeddings
        )

        # 6. Retriever
        # self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        self.retriever = self.vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10}
)

        # 7. Prompt
        self.prompt = ChatPromptTemplate.from_template("""
You are a smart AI assistant.

Answer the question using ONLY the provided context.

Make the response:
- More conversational and natural
- Slightly detailed but not too long
- Well-structured (use bullet points if helpful)
- Add small insights if possible (but DO NOT hallucinate)
- also include some good emojis when necessary and appropriate

If the answer is not in context, say:
"I couldn't find that information in the provided data."

Context:
{context}

Question:
{question}

Answer:
""")

        # 8. Chain
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, query: str):
        return self.chain.invoke(query)