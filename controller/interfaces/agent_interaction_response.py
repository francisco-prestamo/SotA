from entities.sota_table import SotaTable  
from typing import List, Union, Optional
from pydantic import BaseModel

class AgentInteractionResponse(BaseModel):
    text: Optional[str] = None
    sota_table: Optional[SotaTable] = None
    select_list: Optional[List[str]] = None


