""" create a object for the file request """
from pydantic import BaseModel

# process a file if it exists
class FileRequest(BaseModel):
    """ create a file request object """
    name: str
    prompt: str | None = None
