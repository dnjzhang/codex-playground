#!/usr/bin/env python
# Tips from https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300
# Swap out the standard sqllite3 with the latest
import logging


PDF_DIR_PATH = "./pdf-docs"
DB_STORE_PATH = "./db_store"


from langchain_community.embeddings.oci_generative_ai import OCIGenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from load_properties import LoadProperties


# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Loading PDF documents from a directory.
pdf_loader = PyPDFDirectoryLoader(PDF_DIR_PATH)
loaders = [pdf_loader]


documents = []
for loader in loaders:
   documents.extend(loader.load())


text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
all_documents = text_splitter.split_documents(documents)


logging.info(f"Total number of documents: {len(all_documents)}")


#Step 2 - setup OCI Generative AI embedding function
properties = LoadProperties()
logger.info(f"Use auth config profile: {properties.getConfigProfile()}")
embeddings = OCIGenAIEmbeddings(
   model_id=properties.getEmbeddingModelName(),
   service_endpoint=properties.getEndpoint(),
   compartment_id=properties.getCompartment(),
   auth_profile=properties.getConfigProfile(),
   model_kwargs={"truncate":True}
)
#


#Step 3 - Split the documents into patch for embedding due to the embedding model's 96 document limitations.
# Set the batch size
batch_size = 96
# Calculate the number of batches
num_batches = len(all_documents) // batch_size + (len(all_documents) % batch_size > 0)


# Create embeddings and store them to Chroma
db = Chroma(embedding_function=embeddings, persist_directory=(DB_STORE_PATH))
retv = db.as_retriever()


# Iterate over batches
for batch_num in range(num_batches):
   # Calculate start and end indices for the current batch
   start_index = batch_num * batch_size
   end_index = (batch_num + 1) * batch_size
   # Extract documents for the current batch
   batch_documents = all_documents[start_index:end_index]
   # Your code to process each document goes here
   retv.add_documents(batch_documents)
   logging.debug(start_index, end_index)


print("Complete embedding")

