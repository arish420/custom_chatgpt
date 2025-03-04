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
st.write("Hi")

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
st.write(download_db())




agent = create_pandas_dataframe_agent(
    llm, df, agent_type="openai-tools", verbose=True, allow_dangerous_code=True
)

query=st.text_input("Your Query Here")



if st.button("submit"):

    st.write(agent.invoke({"input": query}))











# ###############################################################################################
# files=os.listdir()
# ###############################################################################################
# files=[file for file in files if file.split(".")[-1]=="pdf"]

# ###############################################################################################
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text
# text_data=get_pdf_text(files[:1])




# # # split the extracted data into text chunks using the text_splitter, which splits the text based on the specified number of characters and overlap
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# text_chunks = text_splitter.split_text(text_data)
# # print the number of chunks obtained
# # len(text_chunks)



# modelPath = "BAAI/bge-large-en-v1.5"

# # Create a dictionary with model configuration options, specifying to use the CPU for computations
# model_kwargs = {'device':'cpu'}
# #if using apple m1/m2 -> use device : mps (this will use apple metal)

# # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
# encode_kwargs = {'normalize_embeddings': True}

# # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
# embeddings = HuggingFaceEmbeddings(
#     model_name=modelPath,     # Provide the pre-trained model's path
#     model_kwargs=model_kwargs, # Pass the model configuration options
#     encode_kwargs=encode_kwargs # Pass the encoding options
# )

# # Convert text_chunks (list of strings) to Document objects
# documents = [Document(page_content=chunk) for chunk in text_chunks]  

# # Now use 'documents' instead of 'text_chunks'
# vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)

# # Retrieve and generate using the relevant snippets of the blog.
# retriever = vector_store.as_retriever()

# GROQ_API_KEY=os.getenv("GROQ_API_KEY")

# from langchain_groq import ChatGroq

# llm = ChatGroq(
#     temperature=0,
#     model="mixtral-8x7b-32768",
#     api_key=GROQ_API_KEY
# )

# prompt = hub.pull("rlm/rag-prompt")


# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# st.title("Student Assistant")
# query=""
# query=st.text_input("Write Query Here")

# res=""
# if st.button("Submit") and query!="":
#     res=rag_chain.invoke(query)
#     st.write(res)

#     # # performing a similarity search to fetch the most relevant context
#     st.write("")
#     st.write("")
#     st.write("")

#     context=""

#     for i in vector_store.similarity_search(query):
#         context += i.page_content 

# if res!="":
#     # st.write(context)
#     with st.expander("Feedback"):
#         # Collect user feedback
#         rating = st.slider("Rate this response (1 = Bad, 5 = Excellent)", 1, 5, 3)
#         comment = st.text_area("Any additional feedback?")




#     # Get current date and time
#     # now = datetime.now()
#     # # Format as string
#     # formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
#     # pd.DataFrame({"DateTime":formatted_now,"Context":context,"AI Response":res,"User Feedback":""}.to_excel("user_feedback")
                 
