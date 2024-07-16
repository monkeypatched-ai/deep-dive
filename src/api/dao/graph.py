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


class Machines(StructuredNode):
    """ create a node for the machine metadata """
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    type = StringProperty(unique_index=True, required=True)
    manufacturer = StringProperty(unique_index=True, required=True)
    description = StringProperty(unique_index=True)


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


class SubProcess(StructuredNode):
    """ create a node for the subprocess """
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    description = StringProperty(unique_index=True)
    document = graph.create_relationship_to(Documents, "HAS_A")


class Process(StructuredNode):
    """ create a node for the process  """
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    description = StringProperty(unique_index=True)
    subprocess = graph.create_relationship_to(SubProcess, "HAS_A")
    machine = graph.create_relationship_to(Machines, "HAS_A")


class Subject(StructuredNode):
    """ create a node for the subjects """
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    description = StringProperty(unique_index=True)
    process = graph.create_relationship_to(Process, "HAS_A")


class Topic(StructuredNode):
    """ create a node for the topics """
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    description = StringProperty(unique_index=True)
    subject = graph.create_relationship_to(Subject, "HAS_A")
