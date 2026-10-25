"""The shared LangGraph ToolLoop with a validated final answer (ported from the maxon_dialogue branch, aldev):
an answer that cites unknown sources goes back to the model instead of ending the loop."""
from uuid import uuid4

from langgraph.graph import END
from pydantic import ValidationError

from common.llm.tool_loop import ToolLoop, json_args


class DiscussionLoop(ToolLoop):
    def __init__(self, *args, validate_answer, on_response=None, **kwargs):
        self.validate_answer = validate_answer
        self.on_response = on_response
        super().__init__(*args, **kwargs)

    def _think(self, state):
        if self.on_response is None:
            delta = super()._think(state)
        else:
            from langchain_core.messages import SystemMessage, message_chunk_to_message
            from langchain_core.utils.json import parse_partial_json
            reply = None
            previous = ''
            for chunk in self.model.stream([SystemMessage(self.system_prompt), *state['messages']]):
                reply = chunk if reply is None else reply + chunk
                for call in getattr(reply, 'tool_call_chunks', getattr(reply, 'tool_calls', [])):
                    if call.get('name') != self.final_name:
                        continue
                    try:
                        raw = call.get('args') or '{}'
                        args = parse_partial_json(raw) if isinstance(raw, str) else raw
                    except (ValueError, TypeError):
                        continue
                    text = args.get('response') if isinstance(args, dict) else None
                    if isinstance(text, str) and text != previous:
                        self.on_response(text)
                        previous = text
            if reply is None:
                raise ValueError('Empty model stream')
            delta = {'messages': [message_chunk_to_message(reply)], 'steps': state['steps'] + 1}
        # distinct turns must append even if a provider reuses its message id on a retry
        delta['messages'] = [m.model_copy(update={'id': str(uuid4())}) for m in delta['messages']]
        return delta

    def _validated(self, call):
        result = self.final.model_validate(json_args(self.final, call['args']))
        self.validate_answer(result)
        return result

    def _route(self, state):
        call = self._final_call(state['messages'][-1])
        if call:
            try:
                self._validated(call)
            except (ValidationError, ValueError):
                return END if state['steps'] >= self.max_steps else 'act'
            return END
        return super()._route(state)

    def _execute(self, call):
        if call['name'] == self.final_name:
            try:
                self._validated(call)
            except ValidationError as exc:
                fields = ', '.join('.'.join(map(str, e['loc'])) for e in exc.errors())
                return f'Final answer schema invalid in fields: {fields}. Fix the schema and call {self.final_name} again.'
            except ValueError:
                return 'Unknown source IDs. Cite only sources present in the snapshot or actually returned by tools.'
            return 'Final answer is valid; submit it again without other tool calls.'
        return super()._execute(call)
