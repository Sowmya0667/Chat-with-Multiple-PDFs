import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_openai import ChatOpenAI
import openai


# ------------------ PDF Text Extraction ------------------ #
def get_pdf_text(pdf_docs):
    """Extract text from multiple PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    text += content
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text


# ------------------ Text Chunking ------------------ #
def get_text_chunks(text):
    """Split extracted text into overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


# ------------------ Vectorstore Creation ------------------ #
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_vectorstore(text_chunks):
    model_path = r"C:\Users\sowmy\OneDrive\Desktop\Projects\model\all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(model_name=model_path)

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore



# ------------------ Conversation Chain ------------------ #
def get_conversation_chain(vectorstore, model_name, api_key, api_base): 
    """Create a conversational chain with selected LLM."""
    
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=api_base,
        temperature=0.3,
        max_tokens=512
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        memory=memory
    )
    return conversation_chain



# ------------------ User Interaction ------------------ #
def handle_userinput(user_question):
    """Process user input, perform retrieval + LLM reasoning, and display chat."""
    
    if not st.session_state.conversation:
        st.warning("⚠️ Please upload and process your PDFs first.")
        return

    with st.spinner("🤔 Thinking..."):
        # Get AI response
        response = st.session_state.conversation.invoke({"question": user_question})

        if isinstance(response, dict):
            answer = response.get("answer") or response.get("output_text") or ""
        else:
            answer = str(response)

    if not answer:
        answer = "⚠️ Sorry, I couldn't generate a response. Please try reprocessing your PDFs."

    # Update chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "bot", "content": answer})

    # Display entire chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)



def clear_chat_history():
    """Reset chat memory and UI."""
    st.session_state.chat_history = None
    st.session_state.conversation = None
    st.success("Chat history cleared!")


# ------------------ Streamlit App ------------------ #
def main():
    load_dotenv() 
    
    # --- THIS IS THE FIX ---
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- THIS IS THE OTHER PART OF THE FIX ---
    st.header("Chat with Your PDF Documents :books:")
    
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    
    elif st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


    with st.sidebar:
        st.header("🔑 API Configuration")
        st.markdown("Enter your OpenRouter credentials below. It will fall back to `.env` variables if blank.")
        
        user_api_key = st.text_input(
            "OpenRouter API Key", 
            type="password", 
            placeholder="sk-or-...",
            help="Get your key from https://openrouter.ai/"
        )
        user_api_base = st.text_input(
            "OpenRouter API Base", 
            value="https://openrouter.ai/api/v1",
            help="This is the standard API base for OpenRouter."
        )
        st.markdown("---")
        
        st.subheader("📄 Upload PDFs")
        pdf_docs = st.file_uploader("Upload one or more PDF files", accept_multiple_files=True)

        st.markdown("---")
        
        st.header("🧠 Model Selection")
        selected_model = st.selectbox(
            "Choose a model:",
            [
                "qwen/qwen3-14b:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "meta-llama/llama-3.3-8b-instruct:free"
            ],
            index=0
        )
        
        api_key = user_api_key if user_api_key else os.getenv("OPENROUTER_API_KEY")
        api_base = user_api_base if user_api_base else os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        
        if st.button("🚀 Process"):
            if not api_key:
                st.error("API Key not found. Please add it to the sidebar or your .env file (`OPENROUTER_API_KEY`).")
            elif not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing your PDFs... Please wait."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text:
                            st.error("Could not extract any text from PDFs.")
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            vectorstore = get_vectorstore(text_chunks)
                            st.session_state.conversation = get_conversation_chain(
                                vectorstore, selected_model, api_key, api_base
                            )
                            st.success("Processing complete! You can now chat.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}") 

        st.markdown("---")
        if st.button("🧹 Clear Chat History"):
            clear_chat_history()


if __name__ == "__main__":
    main()