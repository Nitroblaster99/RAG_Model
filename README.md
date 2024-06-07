# RAG_Model
RAG Model for Customer Service

This is an example of one of the use cases of a RAG Model.
It is specifically tuned for customer service.

The RAG technique, or Retrieval Augmented Generation, is an advanced approach to questions and answers that combines elements of information retrieval and natural language generation. It stands out for its efficient use of embeddings and Vector Databases to provide more contextual and informative responses alongside the power of an LLM.

Benefits of the RAG Technique:

1) Advanced Contextualization: The use of embeddings allows the system to understand the semantic context of words, improving the quality of generated responses.
2) Efficient Retrieval: The Vector Databases optimizes the retrieval of relevant information, contributing to more accurate and contextualized answers.
3) Integration of Language Models: The RAG technique integrates natural language generation with information retrieval, delivering more informative and relevant responses.

Most of the code will be built with the use of the Langchain framework, designed to simplify the creation of applications using large language models.
As an embedding model. HuggingFace offers many compatible models in many languages. Most of them are small in size and perfect to be downloaded locally, rather than 
paying for models that have the same efficiency.
As for the vector store, Qdrant is one of the best when it comes to outcomes. It can be used locally or free via their cloud service with an API key.
Lastly, but also very importantly. It is crucial to choose a good and highly trained LLM. Cohere is one the best and also affordable for its services, supporting many languages.

The basic flow of this method is:
Import basic tools.
Upload files. In our case, we will use Markdown files. They are simple, intuitive, compatible, portable, and flexible with all OS.
Split the files into smaller chunks so the model can read and compute efficiently.
Use an embedding model to transform the string files into numerical values.
Pass the embedding model with the now, numerical values. Into a vector store.
Load the LLM we have chosen.
Create a custom prompt template for our use case, giving texture and character to our model.
Create a final chain, that ties everything together.
Ask questions relevant to the data we have loaded into the model.
