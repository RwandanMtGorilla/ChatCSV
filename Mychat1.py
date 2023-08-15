import os
from config import OPENAI_API_KEY, OPENAI_API_BASE
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY 
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
# import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
import csv

# load texts (chunks)

def load_and_split(path: str):
    texts = []
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            texts.append(row[0]) # 假设文本在CSV文件的第一列
    print(f'texts were split into {len(texts)} pieces')
    return texts

#Chroma
embeddings = OpenAIEmbeddings()####
persist_directory = "db"

if not os.path.exists(persist_directory):
    texts = load_and_split("data/data.csv")
    vectordb = Chroma.from_texts(texts=texts, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
else:
    vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

# # #scenario 1: 首次新建向量数据库（只需运行一次）
# texts = load_and_split("data/data.csv") # 替换成你的CSV文件路径
# vectordb = Chroma.from_texts(texts=texts, embedding=embeddings, persist_directory=persist_directory)
# vectordb.persist()
# # # # scenario 2: 已建向量数据库，直接加载
# # vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

# Q&A

# 用streamlit生成web界面
st.title('Chatbot') # 设置标题
user_input = st.text_input('input your qz here') #设置输入框的默认问题

# 根据用户输入，生成回复
if user_input:
    print(f"用户输入：{user_input}")
    # k texts 4
    # fetch_k for MMR texts 20
    # lambda_mult return deviation between texts 0max 1min
    most_relevant_texts = vectordb.max_marginal_relevance_search(user_input, k=4, fetch_k=12, lambda_mult=0)
    # chain_type: stuff（不分段）, map_reduce（分段、分别请求）, refine（分段、依次请求并优化结果，比前者慢）, map-rerank（分段请求，根据分数返回结果）
    #OpenAI(temperature=0.5) #davinci-003#gpt-3.5-turbogpt-4
    llm = OpenAI(temperature=0.5,model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=most_relevant_texts,output_language="Chinese", question="你是一个热心的大三学长 名叫 郝坤 ，正在微信群里解答新生们的问题，你的输入中将包括新生的问题以及数据库中可能和问题有关的信息，以下是输入："+user_input+"请用中文回答")

    st.write(answer)

    # 显示找到的相关度最高的k段文本
    texts_length = 0
    st.write("========================REFERENCES==========================")
    st.write(f"Relevant texts top {len(most_relevant_texts)}:")
    i = 0
    for t in most_relevant_texts:
        i += 1
        st.write(f"********part{i}********")
        st.write(t.page_content)
        texts_length += len(t.page_content)
    print(f"length={texts_length}")
