from neomodel import  StructuredNode, StringProperty,UniqueIdProperty
from api.dao.documents import Documents
from api.dao.documents import graph

class ImageURL(StructuredNode):
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    description = StringProperty(unique_index=True)
    image_url = StringProperty(unique_index=True, required=True)

class Images(Documents):
    imageURL = graph.create_relationship_to(ImageURL, 'HAS_MANY')

    def to_json(self):
        return {
            "uid": self.uid,
            "test": self.name,
            "description": self.description,
            "url": self.imageURL
        }

