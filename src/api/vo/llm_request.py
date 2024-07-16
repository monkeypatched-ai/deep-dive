""" requests for llm  """
from pydantic import BaseModel

class LLMRequest(BaseModel):
    """ create a request to predict the next token"""
    prompt: str
    top_k: str | None = None
    temperature: str | None = None
    top_p: str | None = None


class PretrainRequest(BaseModel):
    """ request for pretraining """
    text: str
