from typing import List, Union, Optional
from pydantic import BaseModel

class SotaTable(BaseModel):
    # Placeholder for SOTA table structure
    columns: List[str]
    rows: List[List[str]]

    @classmethod
    def schema_json(cls, indent: int = 2) -> str:
        """
        Returns the JSON schema for the SotaTable as a string.
        """
        return cls.model_json_schema(indent=indent)

    def __str__(self) -> str:
        """
        Returns a string representation of the SOTA table (columns and rows).
        """
        col_str = " | ".join(self.columns)
        row_strs = [" | ".join(map(str, row)) for row in self.rows]
        table_str = col_str + "\n" + "\n".join(row_strs)
        return table_str