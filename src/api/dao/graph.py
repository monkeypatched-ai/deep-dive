"""  this class is used to create the nodes for the graph """
# pylint: disable=line-too-long,import-error,no-self-use,inconsistent-return-statements,abstract-method

from neomodel import StructuredNode, StringProperty, UniqueIdProperty
from src.db.helpers.graph import GraphDB

graph = GraphDB()

class DocumentChunks(StructuredNode):
    """ create the chunk nodes """
    uid = UniqueIdProperty()
    data = StringProperty(unique_index=True)

    def to_json(self):
        """ convert the result to json"""
        return {"uid": self.uid, "data": self.data}


class Documents(StructuredNode):
    """ create a node for the documents """
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    description = StringProperty(unique_index=True)
    chunks = graph.create_relationship_to(DocumentChunks, "HAS_MANY")

    def to_json(self):
        """ covert the document  metadata to json"""
        return {
            "uid": self.uid,
            "test": self.name,
            "description": self.description,
            "chunks": self.chunks,
        }
        
class Prompt(StructuredNode):
    """ create a node for the prompts """
    uid = UniqueIdProperty()
    text = StringProperty(unique_index=True, required=True)
    description = StringProperty(unique_index=True)
    document = graph.create_relationship_to(Documents, "HAS_MANY")

class Generic(StructuredNode):
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    description = StringProperty(unique_index=True)
