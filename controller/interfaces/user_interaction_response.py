from entities.sota_table import SotaTable  
from typing import List, Union, Optional
from pydantic import BaseModel

class UserInteractionResponse(BaseModel):
    query_text: Optional[str] = None
    selected_element: Optional[str] = None
    selected_list: Optional[List[str]] = None
    sota_table: Optional[SotaTable] = None
