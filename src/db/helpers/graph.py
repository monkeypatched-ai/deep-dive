""" wrapper for the graph API """
# pylint: disable=line-too-long,import-error,not-context-manager,inconsistent-return-statements

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neomodel import config, StructuredNode, RelationshipTo, RelationshipFrom
from neomodel import db
from src.utils.logger import logging as logger

load_dotenv()

URI = str(os.getenv("NEO4J_URL"))

class GraphDB:
    """ helper class around neo4j """
    def __init__(self) -> None:
        """connect to neo4j database"""
        try:
            self.session = None
            self.node = None
            with GraphDatabase.driver(URI) as driver:
                driver.verify_connectivity()
                self.driver = driver
                config.DRIVER = driver
        except ConnectionError as error:
            logger.error("cannot connect to neo4j")
            logger.error(error)

    def create_node(self, node: StructuredNode):
        """create a new node"""
        try:
            self.session = self.driver.session()
            self.node = node
            self.session.close()
            return self.node
        except RuntimeError as error:
            logger.error(f"can not create node with name {self.node.name}")
            logger.error(error)

    def create_relationship_to(self, node: StructuredNode, relationship):
        """create a relationship from the node"""
        try:
            self.session = self.driver.session()
            relationship = RelationshipTo(node, relationship)
            self.session.close()
            return relationship
        except RuntimeError as error:
            logger.error(
                f"can not creeate relationship {relationship} for node {node.name}"
            )
            logger.error(error)

    def create_relation_from(self, node: StructuredNode, relationship):
        """ creates a relationship from the give node to another"""
        try:
            self.session = self.driver.session()
            relationship = RelationshipFrom(node, relationship)
            self.session.close()
            return relationship
        except RuntimeError as error:
            logger.error(
                f"can not creeate relationship {relationship} for node {node.name}"
            )
            logger.error(error)

    def execute_query(self, query):
        """execute the given query from neo4j"""
        try:
            self.session = self.driver.session()
            results, meta = db.cypher_query(query)
            self.session.close()
            return results, meta
        except RuntimeError as error:
            logger.error(f"can not execute given query {query}")
            logger.error(error)

    def add(self, node: StructuredNode):
        """add to data to a paricular node"""
        try:
            self.session = self.driver.session()
            node.save()  # Create
            self.session.close()
            return node
        except RuntimeError as error:
            logger.error("adding data to node")
            logger.error(error)

    def get(self, node: StructuredNode):
        """get the data for the given node"""
        try:
            self.session = self.driver.session()
            all_nodes = node.nodes.all()
            self.session.close()
            return all_nodes
        except RuntimeError as error:
            logger.error("getting data from from node")
            logger.error(error)
