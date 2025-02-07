import os
import openai
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import argparse

# ✅ .env 파일에서 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

def get_hairStyleColor(query):
    if not openai.api_key:
        raise ValueError("🔴 OpenAI API Key가 설정되지 않았습니다. .env 파일을 확인하세요.")

    # 파일에서 URL 목록 읽기
    with open('hairstyleDB.txt', 'r') as f:
        url_list = [line.strip() for line in f.readlines()]

    # 웹 문서 로딩
    loader = WebBaseLoader(web_path=url_list)
    docs = loader.load()

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 벡터 데이터베이스 생성
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # LLM 설정
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0,
        max_tokens=500
    )

    # MultiQueryRetriever 생성
    retriever_multiQuery = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), llm=llm
    )

    # 프롬프트 템플릿
    template = '''Answer the question based only on the following context:
    {context}

    You MUST answer in English.

    Your answer form is:
    hair style: noun, noun

    Question: {question}
    '''
    prompt = ChatPromptTemplate.from_template(template)

    # RAG 체인 구성
    rag_chain = (
        {'context': retriever_multiQuery | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hair style and color suggestion using LLM.")
    parser.add_argument("prompt", type=str, help="Enter a query about hairstyle.")
    args = parser.parse_args()

    result = get_hairStyleColor(args.prompt)
    print(result)