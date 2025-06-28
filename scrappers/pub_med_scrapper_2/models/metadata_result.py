from typing import List
from pydantic import BaseModel
from requests import Response
from . import AuthorSpec
from . import IdSpec


class MetadataResult(BaseModel):
    authors: List[AuthorSpec]
    article_ids: List[IdSpec]

    @staticmethod
    def parse_from_response(pmid: str, response: Response) -> "MetadataResult | None":
        RESULT_FIELD = "result"
        data = response.json()
        uids = data.get(RESULT_FIELD, {}).get("uids", [])
        if not uids:
            return None

        if pmid not in uids:
            return None

        articleids = data[RESULT_FIELD][pmid].get("articleids", [])
        if not articleids: return None
        articleids = MetadataResult._parse_article_ids(articleids)

        authors = data[RESULT_FIELD][pmid].get("authors", [])
        if not authors: return None
        authors = MetadataResult._parse_authors(authors)

        return MetadataResult(
            authors=authors,
            article_ids=articleids
        )



    @staticmethod
    def _parse_article_ids(article_ids: list) -> List[IdSpec]:
        answ = []
        for id in article_ids:
            answ.append(IdSpec.model_validate(id))

        return answ

    @staticmethod
    def _parse_authors(authors: list) -> List[AuthorSpec]:
        answ = []
        for id in authors:
            answ.append(AuthorSpec.model_validate(id))

        return answ
