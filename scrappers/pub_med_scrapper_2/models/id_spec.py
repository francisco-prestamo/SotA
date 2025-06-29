from pydantic import BaseModel


class IdSpec(BaseModel):
    idtype: str
    idtypen: int
    value: str
