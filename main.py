import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
from google.cloud import firestore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredURLLoader, PyMuPDFLoader

load_dotenv()
current_dir = os.getcwd()
persist_dir = os.path.join(current_dir, "db", "chroma_db")
os.makedirs(persist_dir, exist_ok=True)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(current_dir, "/Users/harshithnr/langchain/medical_reasearch_bot/medic-hat-firebase-adminsdk-fbsvc-925b88b753.json")
db_firestore = firestore.Client()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4", temperature=0)

st.set_page_config(page_title="MediChat", layout="wide")
tab_query, tab_logs = st.tabs(["Inquire", "Session logs"])

with tab_query:
    st.title("MediChat 🩺")

    st.sidebar.markdown("### Retrieval Mode")
    restrict_mode = st.sidebar.checkbox("### Strict retrieval only (no extra AI info)")

    st.sidebar.subheader("Add PDF Documents (Limit: 3)")
    pdf_files = st.sidebar.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if pdf_files and len(pdf_files) > 3:
        st.sidebar.warning("Limit of 3 PDF files exceeded.")
        pdf_files = pdf_files[:3]

    st.sidebar.subheader("Add Web Resources (Limit: 3)")
    urls = [st.sidebar.text_input(f"Resource URL {i+1}") for i in range(3)]
    process_clicked = st.sidebar.button("Process Documents")

    if process_clicked:
        valid_urls = [url.strip() for url in urls if url.strip()]
        if not valid_urls and not pdf_files:
            st.warning("Provide at least one input source: PDF or URL.")
        else:
            try:
                all_docs = []
                if os.path.exists(persist_dir):
                    shutil.rmtree(persist_dir)

                if valid_urls:
                    url_loader = UnstructuredURLLoader(urls=valid_urls)
                    url_docs = url_loader.load()
                    for doc in url_docs:
                        doc.metadata["type"] = "URL"
                        doc.metadata["source"] = doc.metadata.get("source", "External Link")
                    all_docs.extend(url_docs)

                for uploaded_pdf in pdf_files or []:
                    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_pdf.read())
                        tmp_path = tmp_file.name
                    pdf_loader = PyMuPDFLoader(tmp_path)
                    pdf_docs = pdf_loader.load()
                    for doc in pdf_docs:
                        doc.metadata["type"] = "PDF"
                        doc.metadata["source"] = f"Uploaded: {uploaded_pdf.name}"
                    all_docs.extend(pdf_docs)

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = splitter.split_documents(all_docs)
                db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
                db.persist()
                st.success("Documents processed successfully.")

            except Exception as e:
                st.error(f"Processing failed: {e}")

    if "history" not in st.session_state:
        st.session_state.history = []

    query = st.text_input("Ask a medical question:")
    if query:
        try:
            if not os.path.exists(persist_dir):
                st.error("No documents have been processed yet.")
                st.stop()

            db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            retriever = db.as_retriever(search_kwargs={"k": 5})

            system_prompt = (
                "You are a medical assistant that ONLY answers based on the provided context. "
                "If the information isn't found in the context, simply say 'I don't know based on cited sources' without any additional explanation."
                if restrict_mode else
                "You are a medical assistant. If the context does not provide a direct answer, start your reply with 'I don't know based on the cited sources.' and then give a brief response from your general knowledge."
            )

            dynamic_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Context: {context}\n\nQuestion: {question}")
            ])

            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                combine_docs_chain_kwargs={
                    "document_prompt": ChatPromptTemplate.from_template("{page_content}"),
                    "document_variable_name": "context",
                    "prompt": dynamic_prompt
                },
                return_source_documents=True
            )

            with st.spinner("Searching documents..."):
                result = chain.invoke({
                    "question": query,
                    "chat_history": st.session_state.history
                })

            if result["source_documents"]:
                response_text = result["answer"]
                sources = list({doc.metadata.get("source", "Unspecified") for doc in result["source_documents"]})
            else:
                response_text = "I don't know." if restrict_mode else "I don't know based on the cited sources. However, here's what I can tell you: " + result["answer"]
                sources = []

            st.session_state.history.append(HumanMessage(content=query))
            st.session_state.history.append(AIMessage(content=response_text))

            st.subheader("Response")
            st.write(response_text)

            if sources:
                st.markdown("**Cited Sources:**")
                for src in sources:
                    st.markdown(f"- [{src}]({src})" if src.startswith("http") else f"- {src}")

            db_firestore.collection("qa_logs").add({
                "question": query,
                "answer": response_text,
                "sources": sources,
                "timestamp": firestore.SERVER_TIMESTAMP
            })

        except Exception as e:
            st.error(f"Query failed: {e}")

with tab_logs:
    st.header("Session Archive")
    try:
        logs = db_firestore.collection("qa_logs").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        for log in logs:
            entry = log.to_dict()
            st.markdown("---")
            st.markdown(f"**Timestamp:** {entry.get('timestamp')}")
            st.markdown(f"**Question:** {entry.get('question')}")
            st.markdown(f"**Answer:** {entry.get('answer')}")
            if entry.get("sources"):
                st.markdown("**Sources:**")
                for src in entry.get("sources", []):
                    st.markdown(f"- [{src}]({src})" if src.startswith("http") else f"- {src}")
    except Exception as e:
        st.error(f"Failed to load session logs: {e}")
