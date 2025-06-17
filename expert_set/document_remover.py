from typing import Dict, List, Tuple
from pydantic import BaseModel
from collections import Counter

from board.board import Board
from entities.sota_table import sota_table_to_dataframe, SotaTable

from .action_picker import PickActionResult
from .prompts.remove_document import (
    DocumentRemovalIntervention,
    create_removal_answers_model,
    remove_document_prompt,
    remove_document_summary_prompt,
    SummaryAnswerModel,
)
from .models import Expert, ExpertDescription
from .interfaces import JsonGenerator


DOCUMENTS_TO_REMOVE = 2  # Maximum number of documents to remove per round


class DocumentRemovalResult(BaseModel):
    documents_removed: List[str]
    summary: str


class DocumentRemover:
    def __init__(
        self,
        json_generator: JsonGenerator,
        board: Board,
        pick_action_result: PickActionResult,
    ):
        self.json_generator = json_generator
        self.index_to_doc_id, self.sota_table_md = (
            self._prepare_sota_table_with_index_mapping(board.sota_table)
        )
        self.thesis_thoughts = self._display_thoughts(board.thesis_knowledge.thoughts)
        self.thesis_description = board.thesis_knowledge.description
        self.expert_presentations = pick_action_result.expert_presentations
        self.board = board

    def _prepare_sota_table_with_index_mapping(
        self, sota_table: SotaTable
    ) -> Tuple[Dict[int, str], str]:
        """
        Creates a dataframe from the SOTA table with document IDs, generates an index-to-document-id mapping,
        and returns both the mapping and the markdown representation of the table.

        Args:
            sota_table: The SOTA table to process

        Returns:
            Tuple containing:
            - Dictionary mapping table indices to document IDs
            - Markdown representation of the table with indices
        """
        df = sota_table_to_dataframe(sota_table, include_id=True)

        index_to_doc_id = {idx: doc_id for idx, doc_id in enumerate(df["id"])}

        df = df.drop(columns=["id"])
        df.insert(0, "Index", list(range(len(df))))

        markdown_table = df.to_markdown(index=False)
        assert markdown_table

        return index_to_doc_id, markdown_table

    def remove_documents(self, experts: List[Expert]) -> DocumentRemovalResult:
        id_to_expert = self._generate_expert_id_dict(experts)
        expert_descriptions = self._extract_descriptions_from_id_dict(id_to_expert)

        answer_model = create_removal_answers_model(expert_descriptions)
        prompt = remove_document_prompt(
            self.expert_presentations,
            self.sota_table_md,
            self.thesis_description,
            self.thesis_thoughts,
            answer_model,
        )

        id_to_intervention = self.json_generator.generate_json(prompt, answer_model)

        documents_to_remove_idxs, documents_to_remove_ids = (
            self._parse_answer_and_get_documents_to_delete(id_to_intervention)
        )

        summary = self._summarize_removal_process(
            prompt, id_to_intervention, documents_to_remove_idxs
        )

        self._remove_documents_from_sota_table_and_update_features(
            documents_to_remove_ids
        )

        return DocumentRemovalResult(
            documents_removed=documents_to_remove_ids,
            summary=summary,
        )

    def _remove_documents_from_sota_table_and_update_features(
        self, document_ids: List[str]
    ):
        to_remove = set(document_ids)
        table_entries = self.board.sota_table.document_features
        feature_list = self.board.sota_table.features

        amount_of_documents_that_have_feature = {f: 0 for f in feature_list}

        for i in reversed(range(len(table_entries))):
            doc, paper_features = table_entries[i]
            if doc.id in to_remove:
                table_entries.pop(i)
            else:
                for feature in paper_features.features.keys():
                    amount_of_documents_that_have_feature[feature] += 1

        features_that_no_document_has = set()
        for feature, amount in amount_of_documents_that_have_feature.items():
            if amount == 0:
                features_that_no_document_has.add(feature)

        for feature in features_that_no_document_has:
            self.board.sota_table.features.remove(feature)

    def _summarize_removal_process(
        self,
        remove_document_prompt: str,
        remove_document_answer: BaseModel,
        documents_removed: List[int],
    ) -> str:
        summary_prompt = remove_document_summary_prompt(
            remove_document_prompt, remove_document_answer, documents_removed
        )

        summary = self.json_generator.generate_json(summary_prompt, SummaryAnswerModel)

        return summary.summary

    def _parse_answer_and_get_documents_to_delete(
        self, id_to_intervention: BaseModel
    ) -> Tuple[List[int], List[str]]:
        vote_counter = Counter()
        for _, intervention in id_to_intervention.model_dump().items():
            intervention = DocumentRemovalIntervention.model_validate(intervention)
            vote_counter.update(intervention.documents_to_delete)

        top_voted_indices = [
            idx for idx, _ in vote_counter.most_common(DOCUMENTS_TO_REMOVE)
        ]

        ids = [self.index_to_doc_id[idx] for idx in top_voted_indices]
        idxs = top_voted_indices

        return idxs, ids

    def _display_thoughts(self, thoughts: List[str]) -> str:
        thesis_thoughts = ""
        for thought in thoughts:
            thesis_thoughts += "- " + thought + "\n"

        return thesis_thoughts

    def _generate_expert_id_dict(self, experts: List[Expert]) -> Dict[str, Expert]:
        answ = {}
        for i, e in enumerate(experts):
            answ["expert_" + str(i)] = e

        return answ

    def _extract_descriptions_from_id_dict(
        self, expert_id_dict: Dict[str, Expert]
    ) -> Dict[str, ExpertDescription]:
        return {id: e.expert_model for id, e in expert_id_dict.items()}
