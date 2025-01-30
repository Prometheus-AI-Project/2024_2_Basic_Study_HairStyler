from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

f = open('hairstyleDB.txt', 'r')
url_list = []

for line in f.readlines():
    url_list.append(line)


loader = WebBaseLoader(web_path=url_list)


docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)



vectorstore = Chroma.from_documents(documents=splits,
                                   embedding=OpenAIEmbeddings())




llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
    max_tokens=500
)



retriever_multiQuery = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm
)

template = '''Answer the question based only on the following context:
{context}

Please involve hair color in your answer

Your answer form is :
hair style: noun, noun, ..
hair color: noun, noun, ..

Question: {question}
'''


prompt = ChatPromptTemplate.from_template(template)



def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)


rag_chain = (
    {'context': retriever_multiQuery | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()       
)


print(rag_chain.invoke("계란형 여자 얼굴에 잘 어울리는 헤어를 알려줘"))
f.close()