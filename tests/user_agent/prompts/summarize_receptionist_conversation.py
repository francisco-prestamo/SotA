from ..models import Chat


def summarize_receptionist_conversation_prompt(receptionist_conversation: Chat) -> str:
    return f"""
You are an expert in summarizing conversations, it follows a conversation between a researcher and
a receptionist of a system specialized in composing the state of the art (SOTA) section of a research paper,
the state of the art section is the section of a research paper that explores similar
investigations on the subject by other researchers, and compares them with their own work. The goal
of the receptionist has been to extract as much information as they can from the user without annoying them.

Your task is to extract the key notes of information about the user's research that was actually extracted,
and to summarize the interaction with the user and the receptionist. Output a simple paragraph, the content of
which will be this summary.

This is the chat:
### CHAT BEGIN ###
{receptionist_conversation.model_dump_json(indent=2)}
### END ###
    """
