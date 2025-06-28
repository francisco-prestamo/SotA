from pydantic import BaseModel


class AuthorSpec(BaseModel):
    name: str
    authtype: str
    clusterid: str
