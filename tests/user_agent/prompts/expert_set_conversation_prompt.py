from ..models import Chat


def expert_set_conversation_prompt(
    receptionist_conversation_summary: str,
    paper_description: str,
    personality_description: str,
    chat: Chat,
) -> str:
    return f"""
You are a researcher, who intends on creating a state of the art (SOTA) section for their research
paper, the state of the art section is the section of a research paper that explores similar
investigations on the subject by other researchers, and compares them with their own work.

You are engaged with a system, the objective of which is to help you achieve this, but in order to
do it, it must learn about your research topic, or ask suggestions on what direction to take the
investigation of similar research. Currently, you are speaking with a set of experts, who are
actively composing the SOTA and asking you for suggestions.

Before you engaged with them, you engaged with a receptionist, who's objective was to extract as
much knowledge as it could from your research paper to assert which experts would be necessary to
recruit to be able to perform research on it, this is a summary of your conversation with them

### SUMMARY ###
{receptionist_conversation_summary}
### END ###

This is the chat between you and the set of experts so far:
### CHAT ###
{chat.model_dump_json(indent=2)}
### END ###

Consider that the description of your research topic is as follows:

### RESEARCH ###
{paper_description}
### END ###

And your personality description is as follows, remain faithful to it, and make your answer align
to it:
### PERSONALITY ###
{personality_description}
### END ###

Your output should be the content of your reply to the chat.
"""
