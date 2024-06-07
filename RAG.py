from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import torch
import cohere
from langchain_cohere import ChatCohere
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import re

# Data loader

# Flatten nested Markdown List

def open_md(file_paths):
    opened_md = []

    for path in file_paths:
        if path.lower().endswith('.md'):
            try:
                md_loader = TextLoader(path)
                md = md_loader.load()
                if isinstance(md, list):
                    opened_md.extend(md)
                else:
                    opened_md.append(md)
            except Exception as e:
                opened_md.append(f"Error opening {path}: {e}")
        else:
            opened_md.append(f"Invalid file type: {path}")

    return opened_md

file_paths = ['',
             '']
opened_md = open_md(file_paths)

def text_splits(data, size: int, overlap: int):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

splits = text_splits(opened_md, 2048, 512)

# Load locally the embedding model that is downloaded from HuggingFace
model_path = r''

# Load the embedding model into cpu
def embedding_model(model_path):
    embeddings = HuggingFaceEmbeddings(model_name=model_path,
                                       model_kwargs={'device': 'cpu'}, encode_kwargs={'device': 'cpu'}, 
                                  )
    return embeddings

embeddings = embedding_model(model_path)

# Load vectorstore and then the retriever
def retriever_function(location):
    vectorstore = Qdrant.from_documents(
    splits,
    embeddings,
    #url=url,
    #prefer_grpc=True,
    #api_key=api_key,
    location=location, 
    #collection_name="my_documents"
    )

    retriever_model = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return retriever_model

retriever = retriever_function(location=":memory:")

# Custom Prompt Template
prompt_template = """"You are a bot for question-answering reviews for a business. \
Use the following pieces of retrieved context to answer the customer's review. \
Do not ask questions in your answer. \
Use we, the pronoun of the first person plural. \
Do not repeat the same phrases and words from the question in your answer. \
Exclude from your answer, words that are used in the question. \
Do not repeat yourself. \
Keep the answer concise.\
Only answer questions related to the business. \
Be less formal. \
Always try to satisfy and acknowledge long-time customers. \
INFORMATION: \n{context}\n
QUESTION: \n{question}\n
ANSWER:  """

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# The printed output sometimes prints \n\n, this replaces any unwanted issues
def clean_string(s: str) -> str:
    """Replace newlines with a blank space in a string."""
    return re.sub(r'\n+', '', s)

def process_result(result):
    """Recursively process and clean strings within the result."""
    if isinstance(result, str):
        return clean_string(result)
    elif isinstance(result, dict):
        return {key: process_result(value) for key, value in result.items()}
    elif isinstance(result, list):
        return [process_result(item) for item in result]
    else:
        return result

# Define the chain and llm
def get_conversation_chain(text:str):
    llm = ChatCohere(cohere_api_key="", model="command-r-plus", max_tokens=512, 
                 temperature=0.1, verbose=True)

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                    retriever=retriever,
                                    chain_type_kwargs={"prompt" : prompt, "verbose": False})

    result = chain.invoke(text)
    
    cleaned_result = process_result(result)
    
    return cleaned_result

# Ask questions!
get_conversation_chain(input())