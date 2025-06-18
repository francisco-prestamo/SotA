from entities.document import Document
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
import pandas as pd

class PaperFeaturesModel(BaseModel):
    authors: List[str]
    title: str
    year: int
    domain: str
    features: Dict[str, Dict[str, Any]]

class SotaTable(BaseModel):
    features: List[str] = Field(default=[])
    document_features: List[Tuple[Document, PaperFeaturesModel]] = Field(default=[])

def sota_table_to_dataframe(sota_table: SotaTable, include_id: bool = False) -> pd.DataFrame:
    rows = []

    for _, paper in sota_table.document_features:
        base_data = {
            "authors": ", ".join(paper.authors),
            "title": paper.title,
            "year": paper.year,
            "domain": paper.domain
        }

        if include_id:
            base_data["id"] = paper.id

        for feature_name in sota_table.features:
            base_data[feature_name] = paper.features.get(feature_name, {}).get("value", None)

        rows.append(base_data)

    df = pd.DataFrame(rows)
    return df


def sota_table_to_markdown(sota_table: SotaTable) -> str:
    df = sota_table_to_dataframe(sota_table)
    return df.to_markdown(index=False)

