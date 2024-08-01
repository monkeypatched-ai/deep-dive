""" main api for the RAG """
# pylint: disable=line-too-long,no-member

import os
import json
import uuid
import socket
import uvicorn
import confluent_kafka
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from llama_index.embeddings.openai import OpenAIEmbedding
# from src.llm.ViT.vit_transformer import VitTransformer
from src.llm.embeddings.image_embeddings import get_image_embedding
from src.db.helpers.qdrant import QdrantDB
from src.lib.initialize_kafka_topics import KafKaTopics
from src.api.dao.graph import graph,SubProcess,Subject,Machines,Process,Topic,DocumentChunks,Documents,Prompt
from src.api.dao.graph_images import Images,ImageURL
from src.api.vo.file_upload_request import FileRequest
from src.api.vo.finetune_request import FinetuneRequest
from src.api.vo.image_upload_request import URLRequest
from src.api.controller.llm import generate_top_k_results
from src.api.vo.llm_request import LLMRequest, PretrainRequest
from src.llm.model.sugriv import sugriv
from src.utils.logger import logging as logger
from src.utils.sematic_splitter import SemanticSplitter
from bin.create_data import create,write
import requests
from PIL import Image
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

KAFKA_SERVER = str(os.getenv("KAFKA_SERVER"))
KAFKA_CONSUMER_GROUP_NAME = str(os.getenv("KAFKA_CONSUMER_GROUP_NAME"))
KAFKA_CONSUMER_OFFSET = str(os.getenv("KAFKA_CONSUMER_OFFSET"))
KAFKA_TOPICS= str(os.getenv("KAFKA_TOPICS"))
MAXIMUM_SEQUENCE_LENGTH = int(os.getenv("MAXIMUM_SEQUENCE_LENGTH"))
D_MODEL = int(os.getenv("D_MODEL"))

sugriv = sugriv.get_model()

# add the api
app = FastAPI()

quadrant = QdrantDB()

# kafka config
KafKaTopics({'bootstrap.servers': KAFKA_SERVER})

producer_conf = {'bootstrap.servers': KAFKA_SERVER,'client.id': socket.gethostname()}
producer = confluent_kafka.Producer(producer_conf)

if quadrant.check_collection_exists("prompts") is not False:
    quadrant.create_collection("prompts")

# create the nodes
SubProcess = graph.create_node(SubProcess)
Machines = graph.create_node(Machines)
Process = graph.create_node(Process)
Subject = graph.create_node(Subject)
Topic = graph.create_node(Topic)
Documents = graph.create_node(Documents)
Prompt = graph.create_node(Prompt)
DocumentChunks = graph.create_node(DocumentChunks)
#create image node
Images = graph.create_node(Images)
ImageURL = graph.create_node(ImageURL)

# add data to nodes
subprocess = graph.add(SubProcess(name="material mixing and master batch",descripton="documents for material mixing"))
machine  = graph.add(Machines(name="MODEL MP1202",type="flow tester", manufacturer="Tinius Olsen",descripton="flow tester/extrusion plastometer by Tinius Olsen"))
process = graph.add(Process(name="gear manufacturing for part p-1290",descripton="documents for gear manufacturing",subprocess=subprocess,machine=machine))
subject = graph.add(Subject(name="injection moulding",descripton="documents for injection moulding",process=process))
topic = graph.add(Topic(name="manufacturing", description="documents for manufacturing processes",subject=subject))

# get the nodes
current_machine = Machines.nodes.get(name="MODEL MP1202")
current_subprocess = SubProcess.nodes.get(name="material mixing and master batch")
current_process = Process.nodes.get(name="gear manufacturing for part p-1290")
current_subject = Subject.nodes.get(name="injection moulding")
current_topic = Topic.nodes.get(name="manufacturing")

# create the relationship between nodes
current_topic.subject.connect(current_subject)
current_subject.process.connect(current_process)
current_process.subprocess.connect(current_subprocess)
current_process.machine.connect(current_machine)

embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            dimensions=D_MODEL
        )

splitter = SemanticSplitter()

# upload a file
@app.post("/upload")
async def create_upload_file(file: UploadFile = File(...)):
    ''' load a document to server'''
    try:
        logger.info(f"loading file {file.filename} to the server")
        with open(file.filename, 'wb') as document:
            content = await file.read()
            document.write(content)
            document.close()
        logger.info(f"loaded file {file.filename} to the server")
        return JSONResponse(content={"filename": file.filename},status_code=200)
    except HTTPException as error:
        logger.error(f'can not load file {file.filename}')
        logger.error(error)
        return JSONResponse(content={"result": "error"},status_code=500)

# create the index
@app.post("/index")
async def create_documents_graph(file:FileRequest):
    '''
    process the document by semantic chunking and adding the data to
    the knowledge graph
        # 1. add the document prompts to vector database
        # 2. add documents chunks to the graph
        # 3. connect the chunks to the document and document metadata
    '''
    try:
        logger.info(f"semantic splitting the given document {file.name} ")
        nodes = splitter.get_nodes([file.name])

        # add the document name as a node to the knowledge graph
        logger.info("creating a node in the graph to store the file name")
        document  = graph.add(Documents(name=file.name,descripton=f"document for {file.name}"))

        # add the prompt as a node to the knowledge graph
        logger.info("creating a node in the graph to add the prompt used to retrieve the document ")
        prompt = graph.add(Prompt(text=file.prompt,descripton=f"prompt for {file.name}"))

        logger.info("connecting the document and prompt to document metadata")
        # create connection from document to prompt and chunks
        prompt.document.connect(document)

        # connect document to the subprocess
        subprocess.document.connect(document)

        logger.info(f"processing the semantic chunks for the document {file.name}")
        # create and save the document chunks
        for i,node in enumerate(nodes):

            logger.info(f"getting text embeddings for the prompt {file.prompt}")
            # add prompt to the vector db
            prompt_vector = embed_model.get_text_embedding(file.prompt)

            logger.info(f"adding prompt {file.prompt} to vector index")
            # add the prompt to the vector database
            quadrant.add(i,{"name" : f"chunk {i}","prompt":str(file.prompt)}, prompt_vector,"prompts")

            # add the document chunk
            logger.info("adding the semantic chunks to the knowledge graph")
            chunk = graph.add(DocumentChunks(name=f"chunk {i}",data=node.get_content()))

            logger.info("connecting the chunks to the documents in the knowledge graph")
            # connect the chunk to the document
            document.chunks.connect(chunk)

        return JSONResponse(content={"filename": file.name},status_code=200)
    except HTTPException as error:
        logger.error(f'can not index file {file.name}')
        logger.error(error)
        return JSONResponse(content={"result": "error"},status_code=500)
    
#create post method for image
@app.post("/upload_image")
async def upload_image(url:URLRequest,desc:str):
    '''
    add an images node to the graph
    add an image url node to the graph
    connect the image url node to the images
    '''
    try:
        #add the image to the qdrant
        logger.info("fetching the image from the url")
        response = requests.get(url.url)
        response.raise_for_status()  # Check if the request was successful

        # Convert the response content to an image
        image = Image.open(BytesIO(response.content))

        # get the image embeddings for the image
        logger.info("creating embeddings for the image")
        image_vector = get_image_embedding(image)

        # add the image to vector db
        quadrant.add({"name" : url.name,"description":desc}, image_vector,name="images")

        # add the image as a node to the knowledge graph
        logger.info(f"creating a node in the graph to store the image name")
        images  = graph.add(Images(name=url.name,description=f"image for {url.name}"))

        # add the ImageURL as a node to the knowledge graph
        logger.info("creating a node in the graph to add the ImageUrl")
        imageURL = graph.add(ImageURL(name=url.name,image_url=url.url,description=f"url for {url.name}"))

        # connect document to the subprocess
        imageURL.images.connect(images)

        return JSONResponse(content={"image_name": url.name},status_code=200)
    except Exception as e:
        logger.error(f'cannot upload image {url.name}')
        logger.error(e)

# feed data to graph rag
@app.post("/feed")
async def feed(request:LLMRequest):
    ''' gets the related documents for the given prompt and then
        feeds them into the LLM'''
    try:
        logger.info("getting documents for prompt")
        # create the vector to retrieve similar prompts
        vector = embed_model.get_text_embedding(request.prompt)

        logger.info("finding top-k similar prompt from the vector database")
        # find similar prompts in the vector database
        similar_prompts = quadrant.get("prompts",vector,top_k=1)

        logger.info("getting the prompt with the similarity score")
        prompt = json.loads(similar_prompts['result'])['payload']['prompt']

        logger.info("lookup the node for the given prompt in the knowledge graph")
        # get the prompt from the graph
        prompt = Prompt.nodes.get(text=prompt)

        logger.info("for the given prompt get the associated documents")
        for document in prompt.document.order_by('name'):

            logger.info("getting the chunks that belong to the document")
            for chunk in document.chunks.order_by('data'):
                uuid4 = uuid.uuid4()
                logger.info("publish the retrieved chunks belonging to the document to enrichment pipeline")
                producer.produce(KAFKA_TOPICS, key=str(uuid4), value=json.dumps(chunk.to_json()))

        return JSONResponse(content={"result": "sent"},status_code=200)
    except HTTPException as error:
        logger.error(f'can not index file {request.prompt}')
        logger.error(error)
        return JSONResponse(content={"result": "error"},status_code=500)

# pretrain the LLM
@app.post("/pretrain")
def pretrain(request:PretrainRequest):
    ''' pretrain llm '''
    try:
        logger.info(request.text)
        sugriv.pretrain_text(request.text)
        return JSONResponse(content={"result": "success"},status_code=200)
    except HTTPException as error:
        logger.error(f'can not pretrain LLM for text file {request.text}')
        logger.error(error)
        return JSONResponse(content={"result": "error"},status_code=500)

@app.post("/create_dataset")
async def create_finetuning_dataset(request:FinetuneRequest):
    ''' create dataset fror finetuning next token prediction '''
    try:
        logger.info('creating dataset for finetuning')
        create(request.texts)
        write('data.json')
        return JSONResponse(content={"result": "success"},status_code=200)
    except HTTPException as error:
        logger.error(f'can not finetune LLM for text {request.texts}')
        logger.error(error)
        return JSONResponse(content={"result": "error"},status_code=500)

# apply top k
@app.post("/search")
async def search(request:LLMRequest):
    ''' generates the top k results using a greedy sampling methodology '''
    try:
        logger.info('getting the top k completions')
        return generate_top_k_results(request)
    except HTTPException as error:
        logger.error('error getting top k completions')
        logger.error(error)
        return JSONResponse(content={"result": "error"},status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
