from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from browser_use.llm.views import ChatInvokeCompletion
from browser_use.qa.compiler import QATaskCompiler, WebUITestCaseDraft, WebUITestStepDraft
from browser_use.qa.judge import judge_test_step, replay_assertion_matches
from browser_use.qa.llm import BrowserUseJudgeTransport, invoke_qa_structured
from browser_use.qa.navigation import NavigationScope
from browser_use.qa.views import (
	ActionCompletionStatus,
	ActionReceipt,
	BrowserEvidenceSnapshot,
	EvidenceArtifact,
	EvidenceKind,
	EvidenceQuality,
	ExpectationSource,
	ExpectationStatus,
	FailureOrigin,
	QAStepStatus,
	ReplayAssertion,
	ReplayAssertionKind,
	StepEvidence,
	StepJudgement,
	StepOperationKind,
	WebUITestStep,
)


def _mock_llm(*completions):
	llm = AsyncMock()
	llm.provider = 'mock'
	llm.model = 'mock-model'
	llm.ainvoke.side_effect = [ChatInvokeCompletion(completion=completion, usage=None) for completion in completions]
	return llm


class _CompiledPayload(BaseModel):
	steps: list[WebUITestStep]


class _RelaxedDraftPayload(BaseModel):
	preconditions: list[str]
	steps: list[dict[str, object]]


def _strict_evidence(*, action_status: ActionCompletionStatus = ActionCompletionStatus.COMPLETED) -> StepEvidence:
	action_artifact = EvidenceArtifact(evidence_id='ev_action', kind=EvidenceKind.ACTION, summary='verified click')
	dom_artifact = EvidenceArtifact(evidence_id='ev_dom', kind=EvidenceKind.DOM, summary='Article editor')
	receipt = ActionReceipt(
		status=action_status,
		operation_kind=StepOperationKind.CLICK,
		tool_succeeded=action_status == ActionCompletionStatus.COMPLETED,
		target_matched=action_status == ActionCompletionStatus.COMPLETED,
		evidence_ids=['ev_action'],
		reasoning='Runner-owned action receipt.',
	)
	return StepEvidence(
		before=BrowserEvidenceSnapshot(url='https://example.com/posts', dom_summary='New button'),
		after=BrowserEvidenceSnapshot(
			url='https://example.com/posts/new',
			dom_summary='Article editor',
			artifacts=[dom_artifact],
		),
		action_receipt=receipt,
		artifacts=[action_artifact],
		evidence_quality=EvidenceQuality.STRONG,
	)


def test_compiler_types_login_preconditions_and_sensitive_references():
	preconditions = QATaskCompiler._typed_preconditions(
		['If the login page is visible, log in with HALO_PASSWORD.'],
		prefix='case_precondition',
	)

	assert preconditions[0].mode == 'ensure'
	assert preconditions[0].sensitive_refs == ['HALO_PASSWORD']


def test_invalid_response_status_assertion_is_a_safe_non_match():
	assert not replay_assertion_matches(
		ReplayAssertion(kind=ReplayAssertionKind.RESPONSE_STATUS, value='not-a-status'),
		BrowserEvidenceSnapshot(url='https://example.com'),
	)


@pytest.mark.asyncio
async def test_chat_browser_use_qa_transport_uses_supported_judgement_schema():
	llm = AsyncMock()
	llm.provider = 'browser-use'
	llm.ainvoke.return_value = ChatInvokeCompletion(
		completion=BrowserUseJudgeTransport(
			reasoning=(
				'{"steps":[{"step_id":"open-articles","instruction":"Click Articles",'
				'"expected_result":"Article list is visible","source_evidence":["Expected article list"]}]}'
			),
			verdict=True,
			failure_reason='',
		),
		usage=None,
	)

	result = await invoke_qa_structured(llm, [], output_format=WebUITestCaseDraft)

	assert result.steps[0].step_id == 'open-articles'
	assert llm.ainvoke.await_args.kwargs['output_format'] is BrowserUseJudgeTransport
	assert llm.ainvoke.await_args.kwargs['request_type'] == 'judge'


@pytest.mark.asyncio
async def test_chat_browser_use_transport_ignores_legacy_envelope_fields_inside_payload():
	llm = AsyncMock()
	llm.provider = 'browser-use'
	llm.ainvoke.return_value = ChatInvokeCompletion(
		completion=BrowserUseJudgeTransport(
			reasoning=(
				'{"steps":[{"step_id":"s1","instruction":"Click New",'
				'"expected_result":"Editor visible","source_evidence":["Expected editor"]}],'
				'"verdict":true,"failure_reason":"","impossible_task":false,"reached_captcha":false}'
			),
			verdict=True,
			failure_reason='',
		),
		usage=None,
	)

	result = await invoke_qa_structured(llm, [], output_format=WebUITestCaseDraft)

	assert result.steps[0].expected_result == 'Editor visible'


@pytest.mark.asyncio
async def test_chat_browser_use_compiler_asks_model_to_repair_missing_evidence():
	llm = AsyncMock()
	llm.provider = 'browser-use'
	llm.ainvoke.side_effect = [
		ChatInvokeCompletion(
			completion=BrowserUseJudgeTransport(
				reasoning=(
					'{"preconditions":[],"steps":[{"step_id":"step_1","instruction":"点击文章",'
					'"expected_result":"显示文章列表","source_evidence":[],"preconditions":[]}]}'
				),
				verdict=True,
				failure_reason='',
			),
			usage=None,
		),
		ChatInvokeCompletion(
			completion=BrowserUseJudgeTransport(
				reasoning=(
					'{"preconditions":[],"steps":[{"step_id":"step_1","instruction":"点击文章",'
					'"expected_result":"显示文章列表","source_evidence":["预期结果：显示文章列表"],'
					'"preconditions":[]}]}'
				),
				verdict=True,
				failure_reason='',
			),
			usage=None,
		),
	]
	compiler = QATaskCompiler(llm)
	task = '打开 https://example.com/console。\n1. 点击文章。预期结果：显示文章列表。'

	draft = await compiler.extract_requirements(task=task)

	assert draft.steps[0].source_evidence == ['预期结果：显示文章列表']
	assert draft.steps[0].requirement_references[0].source == 'task'
	assert task[draft.steps[0].requirement_references[0].start : draft.steps[0].requirement_references[0].end] == (
		'预期结果：显示文章列表'
	)
	assert llm.ainvoke.await_count == 2
	second_messages = llm.ainvoke.await_args_list[1].args[0]
	assert 'previous_output_validation_errors' in str(second_messages[-1].content)


@pytest.mark.asyncio
async def test_chat_browser_use_compiler_asks_model_to_repair_malformed_json():
	llm = AsyncMock()
	llm.provider = 'browser-use'
	valid_payload = (
		'{"preconditions":[],"steps":[{"step_id":"step_1","instruction":"点击文章",'
		'"expected_result":"显示文章列表","source_evidence":["预期结果：显示文章列表"],"preconditions":[]}]}'
	)
	llm.ainvoke.side_effect = [
		ChatInvokeCompletion(
			completion=BrowserUseJudgeTransport(reasoning=f'{valid_payload}}}', verdict=True, failure_reason=''),
			usage=None,
		),
		ChatInvokeCompletion(
			completion=BrowserUseJudgeTransport(reasoning=valid_payload, verdict=True, failure_reason=''),
			usage=None,
		),
	]
	compiler = QATaskCompiler(llm)

	draft = await compiler.extract_requirements(task='打开 https://example.com/console。\n1. 点击文章。预期结果：显示文章列表。')

	assert draft.steps[0].expected_result == '显示文章列表'
	assert llm.ainvoke.await_count == 2
	second_messages = llm.ainvoke.await_args_list[1].args[0]
	assert 'Extra data' in str(second_messages[-1].content)


@pytest.mark.asyncio
async def test_compiler_asks_llm_to_repair_missing_explicit_evidence():
	invalid_draft = _RelaxedDraftPayload(
		preconditions=[],
		steps=[
			{
				'step_id': 'step_1',
				'instruction': '点击文章',
				'expected_result': '显示文章列表',
				'source_evidence': [],
				'preconditions': [],
			}
		],
	)
	repaired_by_llm = WebUITestCaseDraft(
		steps=[
			WebUITestStepDraft(
				step_id='step_1',
				instruction='点击文章',
				expected_result='显示文章列表',
				source_evidence=['预期结果：显示文章列表'],
			)
		]
	)
	llm = _mock_llm(invalid_draft, repaired_by_llm)
	compiler = QATaskCompiler(llm)
	task = '打开 https://example.com/console。\n1. 点击文章。预期结果：显示文章列表。'

	draft = await compiler.extract_requirements(task=task)

	assert draft.steps[0].expected_result == '显示文章列表'
	assert draft.steps[0].source_evidence == ['预期结果：显示文章列表']
	assert llm.ainvoke.await_count == 2
	repair_messages = llm.ainvoke.await_args_list[1].args[0]
	assert 'previous_output_validation_errors' in str(repair_messages[-1].content)


@pytest.mark.asyncio
async def test_compiler_rejects_structure_after_llm_repair_attempts_are_exhausted():
	invalid_draft = _RelaxedDraftPayload(
		preconditions=[],
		steps=[
			{
				'step_id': 'step_1',
				'instruction': 'Click New',
				'expected_result': 'An editor opens',
				'source_evidence': [],
				'preconditions': [],
			}
		],
	)
	llm = _mock_llm(invalid_draft, invalid_draft, invalid_draft)
	compiler = QATaskCompiler(llm)

	with pytest.raises(ValueError, match='after 2 repair attempts'):
		await compiler.extract_requirements(task='Open https://example.com and click New. Expected: an editor opens.')

	assert llm.ainvoke.await_count == 3


@pytest.mark.asyncio
async def test_compiler_preserves_explicit_expectation_and_marks_inferred_step():
	steps = [
		WebUITestStep(
			step_id='login-error',
			instruction='Submit invalid credentials',
			expected_result='The page displays “Invalid password”',
			expectation_source=ExpectationSource.EXPLICIT,
			source_evidence=['预期显示 Invalid password'],
		),
		WebUITestStep(
			step_id='forgot-link',
			instruction='Open the forgot-password flow',
			expected_result='A password reset form is visible',
			expectation_source=ExpectationSource.UI_CONTRACT,
			source_evidence=['Visible link label: Forgot password'],
		),
	]
	draft = WebUITestCaseDraft(
		steps=[
			WebUITestStepDraft(
				step_id='login-error',
				instruction='Submit invalid credentials',
				expected_result='The page displays “Invalid password”',
				source_evidence=['预期显示 Invalid password'],
			),
			WebUITestStepDraft(step_id='forgot-link', instruction='Open the forgot-password flow'),
		]
	)
	compiled_payload = _CompiledPayload(steps=steps)
	llm = _mock_llm(draft, compiled_payload)
	compiler = QATaskCompiler(llm)
	scope = NavigationScope.from_root_url('https://example.com/login')
	case = await compiler.compile(
		task='Open https://example.com/login. Submit invalid credentials; 预期显示 Invalid password。',
		scope=scope,
		discovered_url=scope.root_url,
		discovered_title='Login',
		discovered_dom='Forgot password',
	)
	assert case.steps[0].expectation_source == 'explicit'
	assert case.steps[1].expectation_source == 'ui_contract'
	assert llm.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_all_explicit_expectations_skip_page_discovery_completion():
	draft = WebUITestCaseDraft(
		steps=[
			WebUITestStepDraft(
				step_id='save',
				instruction='Click Save',
				expected_result='A Saved message is visible',
				source_evidence=['Expected: Saved message'],
			)
		]
	)
	llm = _mock_llm(draft)
	case = await QATaskCompiler(llm).compile(
		task='Open https://example.com and click Save. Expected: Saved message.',
		scope=NavigationScope.from_root_url('https://example.com'),
		discovered_url='https://example.com',
		discovered_title='',
		discovered_dom='',
	)
	assert case.steps[0].expectation_source == 'explicit'
	assert llm.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_heuristic_expectation_cannot_be_reported_as_sut_failure():
	step = WebUITestStep(
		step_id='heuristic',
		instruction='Click the tile',
		expected_result='A details page opens',
		expectation_source=ExpectationSource.HEURISTIC,
	)
	judge_output = StepJudgement(
		action_status=ActionCompletionStatus.COMPLETED,
		expectation_status=ExpectationStatus.NOT_MET,
		status=QAStepStatus.SUT_FAILED,
		failure_origin=FailureOrigin.SUT,
		reasoning='The page did not change.',
		actual_result='The same page remains visible.',
		evidence_ids=['ev_dom'],
	)
	evidence = _strict_evidence()
	judgement = await judge_test_step(llm=_mock_llm(judge_output), step=step, evidence=evidence)
	assert judgement.status == QAStepStatus.INCONCLUSIVE
	assert judgement.failure_origin == FailureOrigin.UNKNOWN


@pytest.mark.asyncio
async def test_chat_browser_use_natural_language_positive_verdict_is_inconclusive_in_qa_v2():
	step = WebUITestStep(
		step_id='new-article',
		instruction='Click New',
		expected_result='The editor is visible',
		expectation_source=ExpectationSource.EXPLICIT,
		source_evidence=['Expected editor'],
	)
	evidence = _strict_evidence()
	llm = AsyncMock()
	llm.provider = 'browser-use'
	llm.ainvoke.return_value = ChatInvokeCompletion(
		completion=BrowserUseJudgeTransport(
			reasoning='The New button was clicked and the article editor is visible.',
			verdict=True,
			failure_reason='',
		),
		usage=None,
	)

	judgement = await judge_test_step(llm=llm, step=step, evidence=evidence)

	assert judgement.status == QAStepStatus.INCONCLUSIVE
	assert judgement.failure_origin == FailureOrigin.UNKNOWN
	assert judgement.replay_assertions == []


@pytest.mark.asyncio
async def test_uncompleted_action_receipt_cannot_be_upgraded_to_sut_failure():
	step = WebUITestStep(
		step_id='new-article',
		instruction='Click New',
		expected_result='The editor is visible',
		expectation_source=ExpectationSource.EXPLICIT,
		source_evidence=['Expected editor'],
	)
	evidence = _strict_evidence(action_status=ActionCompletionStatus.NOT_COMPLETED)
	llm = _mock_llm(
		StepJudgement(
			action_status=ActionCompletionStatus.COMPLETED,
			expectation_status=ExpectationStatus.NOT_MET,
			status=QAStepStatus.SUT_FAILED,
			failure_origin=FailureOrigin.SUT,
			reasoning='The editor is absent.',
			actual_result='No editor.',
			evidence_ids=['ev_dom'],
		)
	)

	judgement = await judge_test_step(llm=llm, step=step, evidence=evidence)

	assert judgement.status == QAStepStatus.AGENT_FAILED
	assert judgement.failure_origin == FailureOrigin.AGENT
	assert llm.ainvoke.await_count == 0
