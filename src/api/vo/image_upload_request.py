from typing import List
from pydantic import BaseModel

# process a file if it exists 
class URLRequest(BaseModel):
    name: str
    url: str