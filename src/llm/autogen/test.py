import autogen
from llm_config import claude_haiku

assistant = autogen.ConversableAgent(
    "assistant",
    llm_config={
        "config_list": claude_haiku,
    },
)

user_proxy = autogen.ConversableAgent(
    "user_proxy",
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: x.get("content", "")
    and "TERMINATE" in x.get("content", ""),
    max_consecutive_auto_reply=1,
)


user_proxy.initiate_chat(
    assistant,
    message="Faca um conversos de dolar para real usando 6.15",
)
