import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
import re
import tempfile

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit 페이지 설정
st.set_page_config(page_title="RAG기반 사내 인사규정 QA 시스템", page_icon=":books:", layout="wide")

st.title(":books: 인사담당자와 임직원을 위한 RAG기반 사내 인사규정 QA 시스템")
st.caption("💬안녕하세요! 저는 사내 규정을 알려드리는 친절한 챗봇입니다 :) 원하시는 사내 문서를 왼쪽 사이드바에 첨부하시고, 질문을 해주세요!")

# 상태 초기화
if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processComplete" not in st.session_state:
    st.session_state.processComplete = False

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# 사이드바 설정
with st.sidebar:
    uploaded_files = st.file_uploader("Upload your files", type=['pdf'], accept_multiple_files=True)
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    process = st.button("Process")

    if st.session_state.processComplete:
        st.write("VectorDB was created")
        st.write(f"Document count: {st.session_state.doc_count}")

if process:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    if uploaded_files:
        all_pdf_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            def load_pdf(file_path):
                loader = PyPDFLoader(file_path)
                pdf_docs = loader.load_and_split()
                return pdf_docs

            pdf_docs = load_pdf(tmp_file_path)
            all_pdf_docs.extend(pdf_docs)

        def split_text(page_content):
            split_pattern = r'(?=제\d+조(\의\d+)?\([^)]+\)\s*[^제]*)'
            split_text = re.split(split_pattern, page_content)
            cleaned_split_text = [part.strip() for part in split_text if part and len(part.strip()) > 20]
            return cleaned_split_text

        def extract_metadata(text, page, labor_id, source):
            match = re.search(r'제(\d+조(\의\d+)?)(\([^)]+\))\s*(.+)', text, re.DOTALL)
            if match:
                law_id = int(re.search(r'\d+', match.group(1)).group())
                law_num = f"제{match.group(1).strip()}"
                title = match.group(3).strip('()')
                content = match.group(4).strip()
                return {
                    "labor_id": labor_id,
                    "law_id": law_id,
                    "law_num": law_num,
                    "title": title,
                    "page": page,
                    "source": source,
                    "contents": content
                }
            return None

        metadata = []
        labor_id_counter = 1
        end_of_data = False

        for page_num, page in enumerate(all_pdf_docs):
            if end_of_data:
                break
            page_content = page.page_content
            if '\n부     칙 ' in page_content:
                page_content = page_content.split('\n부     칙 ')[0]
                end_of_data = True
            cleaned_split_text = split_text(page_content)
            page_metadata = []
            for item in cleaned_split_text:
                meta = extract_metadata(item, page_num + 1, labor_id_counter, uploaded_file.name)
                if meta:
                    page_metadata.append(meta)
                    metadata.append(meta)
                    labor_id_counter += 1
            page.metadata = page_metadata

        content_list = []

        for doc in all_pdf_docs:
            for meta in doc.metadata:
                if isinstance(meta, dict):
                    content_list.append((meta['contents'], meta))

        final_docs = []

        for content, meta in content_list:
            new_doc = Document(page_content=content, metadata=meta)
            final_docs.append(new_doc)

        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        st.session_state.vectorstore = FAISS.from_documents(
            documents=final_docs,
            embedding=embeddings_model
        )

        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key),
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2}),
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        )

        st.session_state.processComplete = True
        st.session_state.chat_history = []
        st.session_state.query_input = ""  # 질문 입력 필드 초기화
        st.session_state.doc_count = len(all_pdf_docs)  # 문서의 장수 저장

        # 사이드바에 VectorDB 생성 메시지 및 문서 장수 표시
        with st.sidebar:
            st.write("VectorDB was created")
            st.write(f"Document count: {len(all_pdf_docs)}")

if st.session_state.processComplete:
    def handle_question():
        query = st.session_state.query_input
        response = st.session_state.conversation({"question": query, "chat_history": st.session_state.chat_history})
        answer = response.get("answer", "해당 질문에 대한 답변을 찾을 수 없습니다.")
        st.session_state.chat_history.append((query, answer))
        # 입력 필드 초기화 대신 입력 값을 비워주는 방식으로 수정
        st.session_state.query_input = ""

        # 대화 기록에 추가
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # 대화 기록을 표시
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**🙋 질문:** {message['content']}")
        else:
            st.markdown(f"**🤖 답변:** {message['content']}")

            # 근거 문서 추가
            if message["role"] == "assistant":
                st.markdown("**📄 근거 문서:**")
                retrieved_docs = st.session_state.vectorstore.as_retriever().get_relevant_documents(message["content"])
                for doc in retrieved_docs[:2]:  # 근거 문서 2개만 표시
                    meta = doc.metadata
                    st.markdown(f"**이 규칙은 '{meta['title']}'에 대한 {meta['law_num']}조항입니다 (페이지: {meta['page']}, 출처: {meta['source']})**")
                    st.markdown(doc.page_content[:150])

    # 새로운 질문 입력 필드
    query_input = st.text_input("질문을 입력하세요:", key="query_input", value="")
    st.button("질문 제출", on_click=handle_question)
