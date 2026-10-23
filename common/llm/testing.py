"""Offline stand-in for a chat model: replays scripted AIMessages in order (tool calls included). No network."""
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class ScriptedChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self  # the script already contains the tool calls

    def with_structured_output(self, schema, *, method=None, include_raw=False, **kwargs):
        return super().with_structured_output(schema, include_raw=include_raw)  # fake model knows one method only


_n = 0


def tool_call(name: str, **args) -> dict:
    global _n
    _n += 1
    return {'name': name, 'args': args, 'id': f'call{_n}', 'type': 'tool_call'}


def calls(*tool_calls: dict) -> AIMessage:
    return AIMessage(content='', tool_calls=list(tool_calls))
