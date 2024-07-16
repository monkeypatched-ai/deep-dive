""" create a finetunig request for LLM """
from typing import List
from pydantic import BaseModel


# process a file if it exists
class FinetuneRequest(BaseModel):
    """ create a request for creating the finetuning data """
    texts: List[str]
