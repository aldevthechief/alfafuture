"""Prompts are Markdown files next to the code that uses them: <stage>/<approach>/prompts/<name>.md.

A file is split into chat messages by the header lines `## system` and `## human`; a file without headers is one human
message. `{placeholders}` are template variables, literal braces are written `{{ }}`.
"""
import re
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

ROLE_HEADER = re.compile(r'^## (system|human)\s*$', re.M)


class PromptLibrary:
    def __init__(self, folder: Path | str):
        self.folder = Path(folder)

    def text(self, name: str) -> str:
        return (self.folder / f'{name}.md').read_text(encoding='utf-8').strip()

    def render(self, name: str, **variables) -> str:
        return self.text(name).format(**variables)

    def chat(self, name: str) -> ChatPromptTemplate:
        parts = ROLE_HEADER.split(self.text(name))
        if len(parts) == 1:
            return ChatPromptTemplate.from_messages([('human', parts[0])])
        return ChatPromptTemplate.from_messages([(role, body.strip()) for role, body in zip(parts[1::2], parts[2::2])])

    def names(self) -> list[str]:
        return sorted(p.stem for p in self.folder.glob('*.md'))
