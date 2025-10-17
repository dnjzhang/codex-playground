#!/usr/bin/env python
import argparse
import logging

import chromadb
from chromadb import Settings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings

from load_properties import LoadProperties

# import utils.sqlite3_init

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Use argparse to handle the command-line arguments for the database path, collection name, and query
parser = argparse.ArgumentParser(
    description="Ask question from CPP release notes.")
parser.add_argument("question",
                    help="question to ask the LLM")
parser.add_argument("--top-k", type=int, default=5,
                    help="top K retrieved documents sent to LLM")
parser.add_argument("--temperature", type=float, default=0,
                    help="temperature for the LLM response")

args = parser.parse_args()

# In this demo we will retrieve documents and send these as a context to the LLM.


# Step 1 - setup OCI Generative AI llm


properties = LoadProperties()

logger.info("Connect to [" + properties.getEndpoint() + "] and use model [" + properties.getModelName() + "]");
# use default authN method API-key
llm = ChatOCIGenAI(
    model_id=properties.getModelName(),
    service_endpoint=properties.getEndpoint(),
    compartment_id=properties.getCompartment(),
    model_kwargs={"max_tokens": 2048},
    auth_profile=properties.getConfigProfile()
)

# Step 2 - here we connect to a chromadb server. we need to run the chromadb server before we connect to it


# Alternative when Chroma run on a separate process
# client = chromadb.HttpClient(host="127.0.0.1")


client = chromadb.PersistentClient(
    path="./db_store",
    settings=Settings(),
)

# Step 3 - here we crete embeddings using 'cohere.embed-english-light-v2.0" model.
embeddings = OCIGenAIEmbeddings(
    model_id=properties.getEmbeddingModelName(),
    service_endpoint=properties.getEndpoint(),
    compartment_id=properties.getCompartment(),
    auth_profile=properties.getConfigProfile()
)

# Step 4 - here we create a retriever that gets relevant documents (similar in meaning to a query)


db = Chroma(client=client, embedding_function=embeddings)

retv = db.as_retriever(search_type="similarity", search_kwargs={"k": args.top_k})

# Step 5 - here we can explore how similar documents to the query are returned by prining the document metadata. This step is optional


# user_query = 'Tell us which module is most relevant to LLMs and Generative AI'
user_query = args.question
temperature = args.temperature

docs = retv.invoke(user_query)
print(f"Found number of documents: {len(docs)}")
print(f"Show retrieved document by sending the query:{user_query}")


def pretty_print_docs(docs):
    logger.debug(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


pretty_print_docs(docs)

for doc in docs:
    logger.debug(doc.metadata)

# Step 6 - here we create a retrieval chain that takes llm , retirever objects and invoke it to get a response to our query


logger.debug("Sending the request to LLM along with retrieved documents")
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retv, return_source_documents=True)

# prompt="<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>" + user_query + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
prompt = user_query
response = chain.invoke(prompt, temperature=temperature)

logger.debug(response)

print("Answer:")
print(response['result'])
print("\nSource document references: ")
for doc in response["source_documents"]:
    print(f"Document ID: {doc.metadata}")
