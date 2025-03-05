import streamlit as st
import pandas as pd
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import getpass
import os
from PyPDF2 import PdfReader
import sentence_transformers
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from datetime import datetime
import gdown
import openai

df=pd.read_excel("sample data.xlsx")

from langchain_experimental.agents import create_pandas_dataframe_agent

# https://drive.google.com/file/d/1ug8pf1M1tes-CJMhS_sso372tvC4RQv8/view?usp=sharing

file_id = "1ug8pf1M1tes-CJMhS_sso372tvC4RQv8"
output_file = "open_ai_key.txt"

# Download the file
@st.cache_data
def download_db():
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_file, quiet=False)
    return output_file
k=""
with open(download_db(),'r') as f:
    f=f.read()
    # st.write(f)
    k=f
    
import os
os.environ["OPENAI_API_KEY"] = k
llm = ChatOpenAI(model="gpt-4o-mini")
# st.write(download_db())
json_data=df.to_json(orient='records', indent=4)




from langchain_community.document_loaders import UnstructuredExcelLoader

loader = UnstructuredExcelLoader("sample data.xlsx", mode="elements")
docs = loader.load()

from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Assuming you have OpenAI API key set up in your environment
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)



agent = create_pandas_dataframe_agent(
    llm, df, agent_type="openai-tools", verbose=True, allow_dangerous_code=True
)

st.header("Welcome to Custom AI ChatBot")

query=st.text_input("Your Query Here")



if st.button("submit"):
    # answer = generate_response(query)
    # st.write(answer)


    # st.write(agent.invoke({"input": query}))
    # query = "Who won the Six Nations in 2020?"
    answer = rag_chain.invoke(query)
    st.write(answer)






