""" wrapper around the vector database"""
# pylint: disable=line-too-long,import-error,no-self-use,inconsistent-return-statements

import os
from dotenv import load_dotenv
import requests
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from src.utils.logger import logging as logger

load_dotenv()

QDRANT_HOST = str(os.getenv("QDARANT_HOST"))
QDRANT_PORT = str(os.getenv("QDARANT_PORT"))
VECTOR_SIZE = int(os.getenv("D_MODEL"))

class QdrantDB:
    """ helper classes for qdrant vector database"""

    def __init__(self) -> None:
        """connects to the qdrant vector database"""
        try:
            logger.info(f"connecting to qdrant at {QDRANT_HOST} and port {QDRANT_PORT}")
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except ConnectionError as error:
            logger.error(
                f"error connecting to quadrant db at {QDRANT_HOST} and {QDRANT_PORT}"
            )
            logger.error(error)

    def create_collection(self, name):
        """create a collection in the vector database"""
        try:
            logger.info(f"creating a collection with name {name}")
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            return {"status": "ok"}
        except RuntimeError as error:
            logger.error(error)
            logger.error(f"can not create collection with {name}")
            return {"status": "error"}

    def get(self, name, vector, top_k=5):
        """get a result from the vector databse using the given query"""
        try:
            logger.info(f"getting from collection with name {name} using query")
            search_result = self.client.search(
                collection_name=name, query_vector=vector, limit=top_k
            )
            logger.info("successfully got results from collection")
            return {"status": "ok", "result": search_result[0].json()}
        except RuntimeError as error:
            logger.error(f"error executing query on collection {name}")
            logger.error(error)
            return {"status": "error"}

    def add(self, point_id, payload, vector, name):
        """adds if the id does not exists or else uptates the existing
        recordd for the given collection"""
        try:
            logger.info(f"upserting {str(payload)} into vector database")
            self.client.upsert(
                collection_name=name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        payload=payload,
                        vector=vector,
                    )
                ],
            )
            logger.info("upsert successfull")
        except RuntimeError as error:
            logger.error(f"error upserting {str(payload)} into vector database")
            logger.error(error)

    def check_collection_exists(self, collection_name):
        """ checks if the given collection exists in the vector database"""
        try:
            logger.info("check if client exists")
            if collection_name == "prompts":
                response = requests.get(
                    "http://" + QDRANT_HOST + ":6333/collections/prompts/exists"
                )
                if response.status_code == 404:
                    return False
                return True

            if collection_name == "documents":
                response = requests.get(
                    "http://" + QDRANT_HOST + ":6333/collections/documents/exists"
                )
                if response.status_code == 404:
                    return False
                return True
        except RuntimeError as error:
            logger.error("error checking if collection exists")
            logger.error(error)
            return {"status": "error"}
