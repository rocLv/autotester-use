import re
from unittest.mock import AsyncMock

import pytest

from browser_use import Agent
from browser_use.llm import BaseChatModel
from browser_use.llm.views import ChatInvokeCompletion
from browser_use.qa.compiler import WebUITestCaseDraft
from browser_use.qa.views import QARunStatus, StepJudgement


def _qa_llm() -> BaseChatModel:
	llm = AsyncMock(spec=BaseChatModel)
	llm.model = 'mock-qa-llm'
	llm.model_name = 'mock-qa-llm'
	llm.name = 'mock-qa-llm'
	llm.provider = 'mock'
	llm._verified_api_keys = True
	llm.qa_output_formats = []

	async def invoke(_messages, output_format=None, **_kwargs):
		assert output_format is not None
		llm.qa_output_formats.append(output_format)
		if output_format is WebUITestCaseDraft:
			completion = output_format.model_validate(
				{
					'preconditions': [],
					'steps': [
						{
							'step_id': 'heading',
							'instruction': 'Inspect the main page heading',
							'expected_result': None,
							'source_evidence': [],
						}
					],
				}
			)
		elif output_format.__name__ == '_CompiledSteps':
			completion = output_format.model_validate(
				{
					'steps': [
						{
							'step_id': 'heading',
							'instruction': 'Inspect the main page heading',
							'expected_result': 'The “QA Ready” heading is visible',
							'expectation_source': 'ui_contract',
							'source_evidence': ['Visible h1 text: QA Ready'],
						}
					],
				}
			)
		elif issubclass(output_format, StepJudgement):
			evidence_ids = re.findall(r'ev_[0-9a-f]+', str(_messages))
			assert evidence_ids
			completion = output_format.model_validate(
				{
					'action_status': 'completed',
					'expectation_status': 'met',
					'status': 'PASSED',
					'failure_origin': 'none',
					'reasoning': 'The heading is present in the objective DOM and screenshot evidence.',
					'actual_result': 'The QA Ready heading is visible.',
					'evidence': ['DOM contains heading QA Ready'],
					'evidence_ids': [evidence_ids[0]],
					'confidence': 0.95,
					'replay_assertions': [{'kind': 'dom_contains', 'value': 'QA Ready'}],
				}
			)
		else:
			completion = output_format.model_validate(
				{
					'thinking': 'The requested heading is already observable.',
					'evaluation_previous_goal': '',
					'memory': 'Observed QA Ready heading.',
					'next_goal': 'Submit objective evidence.',
					'action': [
						{
							'finish_test_step': {
								'actual_result': 'The QA Ready heading is visible.',
								'evidence': ['Visible heading text: QA Ready'],
								'action_completed': True,
							}
						}
					],
				}
			)
		return ChatInvokeCompletion(completion=completion, usage=None)

	llm.ainvoke.side_effect = invoke
	return llm


@pytest.mark.asyncio
async def test_default_agent_runs_discovery_judgement_and_returns_typed_qa_result(httpserver, browser_session):
	httpserver.expect_request('/qa').respond_with_data(
		'<!doctype html><html><body><h1>QA Ready</h1><button aria-label="Save changes">Save</button></body></html>',
		content_type='text/html',
	)
	root_url = httpserver.url_for('/qa')
	agent = Agent(
		task=f'Open {root_url}, inspect the main heading, and judge the expected page state.',
		llm=_qa_llm(),
		browser_session=browser_session,
	)
	history = await agent.run(max_steps=3)
	assert history.qa_result is not None
	assert history.qa_result.status == QARunStatus.PASSED
	assert history.qa_result.test_case is not None
	assert history.qa_result.test_case.steps[0].expectation_source == 'ui_contract'
	assert history.is_successful() is True
	# Root navigation is runner-managed and is intentionally absent from model actions.
	assert history.action_names() == ['finish_test_step']


@pytest.mark.asyncio
async def test_rerun_reuses_compiled_case_and_skips_compiler_llm_calls(httpserver, browser_session):
	httpserver.expect_request('/qa-rerun').respond_with_data(
		'<!doctype html><html><body><h1>QA Ready</h1></body></html>',
		content_type='text/html',
	)
	root_url = httpserver.url_for('/qa-rerun')
	llm = _qa_llm()
	agent = Agent(
		task=f'Open {root_url}, inspect the main heading, and judge the expected page state.',
		llm=llm,
		browser_session=browser_session,
	)

	first_history = await agent.run(max_steps=3)
	assert first_history.qa_result is not None
	assert first_history.qa_result.status == QARunStatus.PASSED
	first_compiler_calls = getattr(llm, 'qa_output_formats').count(WebUITestCaseDraft)
	assert first_compiler_calls == 1

	second_history = await agent.rerun(max_steps=3)

	assert second_history.qa_result is not None
	assert second_history.qa_result.status == QARunStatus.PASSED
	second_compiler_calls = getattr(llm, 'qa_output_formats').count(WebUITestCaseDraft)
	assert second_compiler_calls == first_compiler_calls


@pytest.mark.asyncio
async def test_replay_rerun_uses_no_llm_calls(httpserver, browser_session):
	httpserver.expect_request('/qa-zero-llm').respond_with_data(
		'<!doctype html><html><body><h1>QA Ready</h1></body></html>',
		content_type='text/html',
	)
	root_url = httpserver.url_for('/qa-zero-llm')
	llm = _qa_llm()
	agent = Agent(
		task=f'Open {root_url}, inspect the “QA Ready” heading.',
		llm=llm,
		browser_session=browser_session,
	)

	first_history = await agent.run(max_steps=3)
	assert first_history.qa_result is not None
	assert first_history.qa_result.status == QARunStatus.PASSED
	llm_calls_after_first_run = len(getattr(llm, 'qa_output_formats'))

	replay_history = await agent.rerun(mode='replay')

	assert replay_history.qa_result is not None
	assert replay_history.qa_result.status == QARunStatus.PASSED
	assert len(getattr(llm, 'qa_output_formats')) == llm_calls_after_first_run
	phase_names = {timing.phase for timing in replay_history.qa_result.phase_timings}
	assert {'browser_start', 'root_navigation', 'replay_total'} <= phase_names


@pytest.mark.asyncio
async def test_bundle_replay_works_in_a_fresh_agent_with_strict_zero_llm(httpserver, browser_session, tmp_path):
	httpserver.expect_request('/qa-bundle').respond_with_data(
		'<!doctype html><html><body><h1>QA Ready</h1></body></html>',
		content_type='text/html',
	)
	root_url = httpserver.url_for('/qa-bundle')
	task = f'Open {root_url}, inspect the “QA Ready” heading.'
	first_agent = Agent(task=task, llm=_qa_llm(), browser_session=browser_session)
	first_history = await first_agent.run(max_steps=3)
	assert first_history.qa_result is not None
	assert first_history.qa_result.status == QARunStatus.PASSED
	bundle = first_agent.save_qa_bundle(tmp_path / 'bundle')

	replay_llm = _qa_llm()
	second_agent = Agent(task=task, llm=replay_llm, browser_session=browser_session)
	replay_history = await second_agent.rerun(mode='replay', bundle=bundle.path, allow_llm_fallback=False)

	assert replay_history.qa_result is not None
	assert replay_history.qa_result.status == QARunStatus.PASSED
	assert replay_history.qa_result.requested_mode == 'replay'
	assert replay_history.qa_result.effective_mode == 'replay'
	assert replay_history.qa_result.llm_call_count == 0
	assert getattr(replay_llm, 'qa_output_formats') == []
