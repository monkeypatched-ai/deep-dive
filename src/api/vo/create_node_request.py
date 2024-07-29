from pydantic import BaseModel


class CreateNodeRequest(BaseModel):
    node_type: str
    node_name: str
    node_description: str
    set_relationship: bool = False
    current_node_type: str | None = None
    current_node_name: str | None = None
