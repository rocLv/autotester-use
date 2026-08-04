"""Structured LLM transport helpers for QA compilation and judgement."""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, Field

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import BaseMessage, SystemMessage

QAOutput = TypeVar('QAOutput', bound=BaseModel)


class BrowserUseJudgeTransport(BaseModel):
	"""Schema required by ChatBrowserUse's dedicated judge endpoint."""

	reasoning: str | None = Field(default=None, description='Explanation of the judgement')
	verdict: bool = Field(description='Whether the trace was successful or not')
	failure_reason: str | None = Field(
		default=None,
		description='Explanation of why serialization failed; empty when verdict is true',
	)
	impossible_task: bool = False
	reached_captcha: bool = False


class BrowserUseStructuredPayloadError(ValueError):
	"""Raised when the judge transport returns a verdict but omits the JSON payload."""

	def __init__(self, message: str, *, transport: BrowserUseJudgeTransport):
		super().__init__(message)
		self.transport = transport


def _parse_json_object(value: str) -> object:
	"""Parse a JSON object, tolerating an accidental Markdown code fence."""

	text = value.strip()
	if text.startswith('```'):
		lines = text.splitlines()
		if len(lines) >= 3 and lines[-1].strip() == '```':
			text = '\n'.join(lines[1:-1])
	start = text.find('{')
	end = text.rfind('}')
	if start < 0 or end < start:
		raise ValueError('ChatBrowserUse QA transport did not return a JSON object in reasoning')
	return json.loads(text[start : end + 1])


async def invoke_qa_structured(
	llm: BaseChatModel,
	messages: list[BaseMessage],
	*,
	output_format: type[QAOutput],
) -> QAOutput:
	"""Invoke any supported model with a Pydantic QA result schema.

	ChatBrowserUse's dedicated judge endpoint accepts only ``JudgementResult``.
	For that provider, the required QA model is carried as JSON inside the typed
	``reasoning`` field and validated again locally. Other providers receive the
	QA Pydantic model directly.
	"""

	if llm.provider != 'browser-use':
		response = await llm.ainvoke(messages, output_format=output_format)
		completion = response.completion
		if isinstance(completion, output_format):
			return completion
		if isinstance(completion, BaseModel):
			completion = completion.model_dump()
		return output_format.model_validate(completion)

	schema = json.dumps(output_format.model_json_schema(), ensure_ascii=False)
	transport_instruction = SystemMessage(
		content=f"""The Browser Use judge transport requires JudgementResult, but the caller requires a different typed QA object.
Put ONLY one compact JSON object matching <qa_output_schema> in the `reasoning` field.
Set `verdict` true when serialization succeeded and `failure_reason` to an empty string.
Do not wrap the JSON in Markdown and do not omit required fields.
<qa_output_schema>{schema}</qa_output_schema>"""
	)
	response = await llm.ainvoke(
		[messages[0], transport_instruction, *messages[1:]] if messages else [transport_instruction],
		output_format=BrowserUseJudgeTransport,
		request_type='judge',
	)
	transport = BrowserUseJudgeTransport.model_validate(response.completion)
	if not transport.reasoning:
		raise ValueError(f'ChatBrowserUse QA transport returned no structured reasoning: {transport.failure_reason}')
	try:
		payload = _parse_json_object(transport.reasoning)
	except ValueError as exc:
		raise BrowserUseStructuredPayloadError(
			f'{exc}; received: {transport.reasoning[:500]}',
			transport=transport,
		) from exc
	if isinstance(payload, dict):
		# The dedicated judge occasionally appends its own transport fields to the
		# requested QA object. They carry no QA semantics and must not make an
		# otherwise valid, strongly typed payload fail ``extra='forbid'`` validation.
		payload = {key: value for key, value in payload.items() if key in output_format.model_fields}
	return output_format.model_validate(payload)
