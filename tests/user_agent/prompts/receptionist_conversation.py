from ..models import Chat


def receptionist_conversation_prompt(
    chat: Chat, paper_description: str, personality_description: str
) -> str:
    return f"""
You are a researcher, who intends on creating a state of the art (SOTA) section for their research
paper, the state of the art section is the section of a research paper that explores similar
investigations on the subject by other researchers, and compares them with their own work.

You are engaged with a system, the objective of which is to help you achieve this, but in order to
do it, it must learn about your research topic. Currently, you are speaking to a receptionist of that
system, that is trying to discern all the information it can about your research.

This is your chat so far with the receptionist:
### BEGIN ###
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
