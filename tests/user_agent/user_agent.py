from numpy import roll
from receptionist_agent.interfaces import UserAPI
from expert_set.interfaces import UserQuerier

from .models import Chat, ChatEntry, SimpleTextAnswerModel
from .prompts import (
    receptionist_conversation_prompt,
    summarize_receptionist_conversation_prompt,
    expert_set_conversation_prompt,
)
from .interfaces import JsonGenerator


class UserAgent(UserAPI, UserQuerier):
    def __init__(
        self,
        paper_description: str,
        personality_description: str,
        json_generator: JsonGenerator,
    ):
        self.receptionist_chat = Chat(entries=[])
        self.expert_set_chat = Chat(entries=[])
        self.paper_description = paper_description
        self.personality_description = personality_description
        self.json_generator = json_generator

        self.receptionist_conversation_summary: str | None = None

    def receptionist_query_user(self, query: str) -> str:
        self._add_receptionist_intervention_to_chat(query)
        prompt = receptionist_conversation_prompt(
            self.receptionist_chat, self.paper_description, self.personality_description
        )
        answ = self.json_generator.generate_json(prompt, SimpleTextAnswerModel)
        answ = answ.content
        self._add_researcher_intervention_to_chat(answ, self.receptionist_chat)
        return answ

    def receptionist_message_user(self, text: str) -> None:
        self._add_receptionist_intervention_to_chat(text)

    def expert_set_query_user(self, query: str) -> str:
        self._add_expert_set_intervention_to_chat(query)

        receptionist_conversation_summary = (
            self._get_receptionist_conversation_summary()
        )
        prompt = expert_set_conversation_prompt(
            receptionist_conversation_summary,
            self.paper_description,
            self.personality_description,
            self.expert_set_chat,
        )
        answer = self.json_generator.generate_json(prompt, SimpleTextAnswerModel)
        answer = answer.content

        self._add_researcher_intervention_to_chat(answer, self.expert_set_chat)

        return answer

    def _get_receptionist_conversation_summary(self) -> str:
        if not self.receptionist_conversation_summary:
            self.receptionist_conversation_summary = (
                self._compute_receptionist_conversation_summary()
            )
        return self.receptionist_conversation_summary

    def _compute_receptionist_conversation_summary(self) -> str:
        prompt = summarize_receptionist_conversation_prompt(self.receptionist_chat)
        answer = self.json_generator.generate_json(prompt, SimpleTextAnswerModel)
        answer = answer.content
        return answer

    def _add_receptionist_intervention_to_chat(self, intervention: str) -> None:
        self.receptionist_chat.entries.append(
            ChatEntry(role="Receptionist", content=intervention)
        )

    def _add_researcher_intervention_to_chat(
        self, intervention: str, chat: Chat
    ) -> None:
        chat.entries.append(ChatEntry(role="Researcher", content=intervention))

    def _add_expert_set_intervention_to_chat(self, intervention: str) -> None:
        self.expert_set_chat.entries.append(
            ChatEntry(role="Expert Set", content=intervention)
        )

