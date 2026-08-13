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
	FailureCode,
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


class _RelaxedJudgementPayload(BaseModel):
	action_status: str
	expectation_status: str
	status: str
	failure_origin: str
	failure_code: str = 'none'
	reasoning: str
	actual_result: str
	evidence_ids: list[str] = []
	confidence: float = 0.5
	retry_safe: bool = False


def _strict_evidence(
	*,
	action_status: ActionCompletionStatus = ActionCompletionStatus.COMPLETED,
	target_matched: bool | None = True,
	tool_succeeded: bool = True,
	evidence_quality: EvidenceQuality = EvidenceQuality.STRONG,
) -> StepEvidence:
	action_artifact = EvidenceArtifact(evidence_id='ev_action', kind=EvidenceKind.ACTION, summary='verified click')
	dom_artifact = EvidenceArtifact(evidence_id='ev_dom', kind=EvidenceKind.DOM, summary='Article editor')
	receipt = ActionReceipt(
		status=action_status,
		operation_kind=StepOperationKind.CLICK,
		tool_succeeded=tool_succeeded,
		target_matched=target_matched,
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
		evidence_quality=evidence_quality,
	)


def test_compiler_types_login_preconditions_and_sensitive_references():
	preconditions = QATaskCompiler._typed_preconditions(
		['If the login page is visible, log in with HALO_PASSWORD.'],
		prefix='case_precondition',
	)

	assert preconditions[0].mode == 'ensure'
	assert preconditions[0].sensitive_refs == ['HALO_PASSWORD']


def test_stage_one_schema_requires_source_evidence_from_the_model():
	step_schema = WebUITestStepDraft.model_json_schema()

	assert 'source_evidence' in step_schema['required']


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
async def test_compiler_ignores_model_authored_provenance_and_derives_verified_span():
	task = '打开 https://example.com/console。点击文章。预期结果：显示文章列表。'
	llm = AsyncMock()
	llm.provider = 'browser-use'
	llm.ainvoke.return_value = ChatInvokeCompletion(
		completion=BrowserUseJudgeTransport(
			reasoning=(
				'{"preconditions":[],"steps":[{"step_id":"step_1","instruction":"点击文章",'
				'"expected_result":"显示文章列表","source_evidence":["预期结果：显示文章列表"],'
				'"requirement_references":[{"source":"task","quote":"预期结果：显示文章列表"}],'
				'"preconditions":[]}]}'
			),
			verdict=True,
			failure_reason='',
		),
		usage=None,
	)

	draft = await QATaskCompiler(llm).extract_requirements(task=task)

	reference = draft.steps[0].requirement_references[0]
	assert llm.ainvoke.await_count == 1
	assert reference.source == 'task'
	assert reference.start is not None
	assert reference.end is not None
	assert task[reference.start : reference.end] == reference.quote


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
async def test_compiler_asks_model_to_repair_non_verbatim_requirement_quote():
	task = '打开 https://example.com/console。点击“文章”。预期结果：页面展示全部文章。'
	invalid_quote = _RelaxedDraftPayload(
		preconditions=[],
		steps=[
			{
				'step_id': 'step_1',
				'instruction': '点击文章',
				'expected_result': '显示文章列表',
				'source_evidence': ['点击‘文章’'],
				'preconditions': [],
			}
		],
	)
	repaired = WebUITestCaseDraft(
		steps=[
			WebUITestStepDraft(
				step_id='step_1',
				instruction='点击文章',
				expected_result='页面展示全部文章',
				source_evidence=['预期结果：页面展示全部文章'],
			)
		]
	)
	llm = _mock_llm(invalid_quote, repaired)

	draft = await QATaskCompiler(llm).extract_requirements(task=task)

	assert llm.ainvoke.await_count == 2
	assert draft.steps[0].requirement_references[0].quote == '预期结果：页面展示全部文章'
	repair_messages = llm.ainvoke.await_args_list[1].args[0]
	assert 'not an exact Task/ground_truth quote' in str(repair_messages[-1].content)


@pytest.mark.asyncio
async def test_compiler_uses_exact_expected_result_when_model_normalizes_evidence_punctuation():
	task = '打开 https://example.com/console。点击“文章”。预期结果：显示文章列表。'
	model_payload = _RelaxedDraftPayload(
		preconditions=[],
		steps=[
			{
				'step_id': 'step_1',
				'instruction': '点击文章',
				'expected_result': '显示文章列表',
				'source_evidence': ['点击‘文章’'],
				'preconditions': [],
			}
		],
	)
	llm = _mock_llm(model_payload)

	draft = await QATaskCompiler(llm).extract_requirements(task=task)

	assert llm.ainvoke.await_count == 1
	assert draft.steps[0].requirement_references[0].quote == '显示文章列表'


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
			WebUITestStepDraft(step_id='forgot-link', instruction='Open the forgot-password flow', source_evidence=[]),
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
	stage_two_system_prompt = str(llm.ainvoke.await_args_list[1].args[0][0].content)
	assert 'Browser Use Website Functionality Testing method' in stage_two_system_prompt
	assert 'For null expected results only' in stage_two_system_prompt


@pytest.mark.asyncio
async def test_compiler_uses_functionality_testing_guidance_without_expanding_scope():
	draft = WebUITestCaseDraft(
		steps=[
			WebUITestStepDraft(
				step_id='step_1',
				instruction='Click Articles',
				expected_result='The article list is visible',
				source_evidence=['Expected: the article list is visible'],
			)
		]
	)
	llm = _mock_llm(draft)

	result = await QATaskCompiler(llm).extract_requirements(
		task='Open https://example.com. Click Articles. Expected: the article list is visible.'
	)

	system_prompt = str(llm.ainvoke.await_args.args[0][0].content)
	assert 'Browser Use Website Functionality Testing method' in system_prompt
	assert 'UI elements, data entry and validation, error handling and messaging' in system_prompt
	assert 'coverage examples, not default requirements' in system_prompt
	assert 'Actual behavior, issue screenshots, and severity are runtime evidence' in system_prompt
	assert 'Set expected_result only when it is explicitly stated' in system_prompt
	assert [step.step_id for step in result.steps] == ['step_1']


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
async def test_heuristic_expectation_can_be_reported_as_sut_failure_from_facts():
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
	assert judgement.status == QAStepStatus.SUT_FAILED
	assert judgement.failure_origin == FailureOrigin.SUT


@pytest.mark.asyncio
async def test_judge_llm_schema_rejects_inconclusive_status():
	step = WebUITestStep(
		step_id='new-article',
		instruction='Click New',
		expected_result='The editor is visible',
		expectation_source=ExpectationSource.EXPLICIT,
		source_evidence=['Expected editor'],
	)
	llm = _mock_llm(
		StepJudgement(
			action_status=ActionCompletionStatus.COMPLETED,
			expectation_status=ExpectationStatus.MET,
			status=QAStepStatus.PASSED,
			failure_origin=FailureOrigin.NONE,
			reasoning='The editor is visible.',
			actual_result='Article editor visible.',
			evidence_ids=['ev_dom'],
		)
	)

	judgement = await judge_test_step(llm=llm, step=step, evidence=_strict_evidence())
	output_format = llm.ainvoke.await_args.kwargs['output_format']

	assert judgement.status == QAStepStatus.PASSED
	assert 'INCONCLUSIVE' in output_format.model_json_schema()['$defs']['QAStepStatus']['enum']

	with pytest.raises(ValueError, match='concrete status'):
		output_format(
			action_status=ActionCompletionStatus.COMPLETED,
			expectation_status=ExpectationStatus.NOT_OBSERVABLE,
			status=QAStepStatus.INCONCLUSIVE,
			failure_origin=FailureOrigin.UNKNOWN,
			reasoning='I cannot decide.',
			actual_result='Unknown.',
		)


@pytest.mark.asyncio
async def test_judge_accepts_concrete_verdict_without_evidence_ids():
	step = WebUITestStep(
		step_id='new-article',
		instruction='Click New',
		expected_result='The editor is visible',
		expectation_source=ExpectationSource.EXPLICIT,
		source_evidence=['Expected editor'],
	)
	missing_citation = StepJudgement(
		action_status=ActionCompletionStatus.COMPLETED,
		expectation_status=ExpectationStatus.MET,
		status=QAStepStatus.PASSED,
		failure_origin=FailureOrigin.NONE,
		reasoning='The editor is visible.',
		actual_result='Article editor visible.',
	)
	llm = _mock_llm(missing_citation)
	call_count = 0

	def count_call() -> None:
		nonlocal call_count
		call_count += 1

	judgement = await judge_test_step(
		llm=llm,
		step=step,
		evidence=_strict_evidence(),
		on_llm_call=count_call,
	)

	assert judgement.status == QAStepStatus.PASSED
	assert judgement.evidence_ids == []
	assert llm.ainvoke.await_count == 1
	assert call_count == 1


@pytest.mark.asyncio
async def test_judge_repairs_repeated_schema_invalid_outputs_with_llm():
	step = WebUITestStep(
		step_id='publish',
		instruction='Click Publish in the article settings modal',
		expected_result='The article appears in the management list and on the homepage.',
		expectation_source=ExpectationSource.EXPLICIT,
		source_evidence=['Expected article appears in both places'],
	)
	evidence = _strict_evidence(action_status=ActionCompletionStatus.COMPLETED)
	http_502_artifact = EvidenceArtifact(evidence_id='ev_http_502', kind=EvidenceKind.NETWORK, summary='HTTP 502 document')
	evidence.after = BrowserEvidenceSnapshot(
		url='https://demo.halocms.site/console/posts',
		dom_summary='HTTP ERROR 502',
		network_errors=['HTTP 502 [Document] https://demo.halocms.site/console/posts'],
		artifacts=[http_502_artifact],
	)
	evidence.artifacts.append(http_502_artifact)
	invalid_output = _RelaxedJudgementPayload(
		action_status='not_completed',
		expectation_status='not_met',
		status='SUT_FAILED',
		failure_origin='sut',
		failure_code='sut_related_http_error',
		reasoning='The page returned HTTP 502.',
		actual_result='HTTP 502 prevented verification.',
		evidence_ids=['ev_http_502'],
	)
	repaired_output = StepJudgement(
		action_status=ActionCompletionStatus.COMPLETED,
		expectation_status=ExpectationStatus.NOT_MET,
		status=QAStepStatus.SUT_FAILED,
		failure_origin=FailureOrigin.SUT,
		failure_code=FailureCode.SUT_RELATED_HTTP_ERROR,
		reasoning='The publish action was attempted, but the expected article visibility was absent after HTTP 502.',
		actual_result='The page shows HTTP ERROR 502 instead of the article list or homepage article.',
		evidence_ids=['ev_http_502'],
	)
	llm = _mock_llm(invalid_output, invalid_output, repaired_output)

	judgement = await judge_test_step(llm=llm, step=step, evidence=evidence)

	assert judgement.status == QAStepStatus.SUT_FAILED
	assert judgement.failure_code == FailureCode.SUT_RELATED_HTTP_ERROR
	assert judgement.evidence_ids == ['ev_http_502']
	assert llm.ainvoke.await_count == 3
	repair_messages = llm.ainvoke.await_args_list[1].args[0]
	assert 'previous structured judgement was rejected' in str(repair_messages[-1].content)
	assert 'SUT_FAILED requires objective evidence that the action completed' in str(repair_messages[-1].content)


@pytest.mark.asyncio
async def test_judge_uses_llm_even_when_custom_tool_expectation_exists():
	step = WebUITestStep(
		step_id='slider',
		instruction='Complete the slider',
		expected_result='The slider displays verification successful.',
		expectation_source=ExpectationSource.EXPLICIT,
		source_evidence=['预期结果：验证成功'],
	)
	action_artifact = EvidenceArtifact(
		evidence_id='ev_slider_action',
		kind=EvidenceKind.ACTION,
		summary='Slider success checked by custom tool',
		metadata={
			'tool_expectation_proofs': [
				{
					'requirement_quote': '滑块区域明确显示“验证成功”。',
					'expectation_met': True,
					'verification': {'site_success_observed': True},
				}
			]
		},
	)
	receipt = ActionReceipt(
		status=ActionCompletionStatus.COMPLETED,
		operation_kind=StepOperationKind.OTHER,
		tool_succeeded=True,
		target_matched=True,
		evidence_ids=['ev_slider_action'],
		reasoning='The slider action completed.',
	)
	evidence = StepEvidence(
		before=BrowserEvidenceSnapshot(url='https://example.com/register', dom_summary='请拖动滑块'),
		after=BrowserEvidenceSnapshot(url='https://example.com/register', dom_summary='验证成功'),
		action_receipt=receipt,
		artifacts=[action_artifact],
		evidence_quality=EvidenceQuality.STRONG,
	)
	llm = _mock_llm(
		StepJudgement(
			action_status=ActionCompletionStatus.COMPLETED,
			expectation_status=ExpectationStatus.MET,
			status=QAStepStatus.PASSED,
			failure_origin=FailureOrigin.NONE,
			reasoning='The final page displays verification successful.',
			actual_result='Verification success is visible.',
			evidence_ids=['ev_slider_action'],
		)
	)

	judgement = await judge_test_step(llm=llm, step=step, evidence=evidence)

	assert judgement.status == QAStepStatus.PASSED
	assert judgement.evidence_ids == ['ev_slider_action']
	assert llm.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_chat_browser_use_natural_language_positive_verdict_is_passed():
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

	assert judgement.status == QAStepStatus.PASSED
	assert judgement.failure_origin == FailureOrigin.NONE


@pytest.mark.asyncio
async def test_action_receipt_does_not_gate_fact_based_sut_failure():
	step = WebUITestStep(
		step_id='new-article',
		instruction='Click New',
		expected_result='The editor is visible',
		expectation_source=ExpectationSource.EXPLICIT,
		source_evidence=['Expected editor'],
	)
	evidence = _strict_evidence(action_status=ActionCompletionStatus.NOT_COMPLETED, target_matched=False, tool_succeeded=False)
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

	assert judgement.status == QAStepStatus.SUT_FAILED
	assert judgement.failure_origin == FailureOrigin.SUT
	assert llm.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_judge_passes_from_facts_when_element_target_is_not_verified():
	step = WebUITestStep(
		step_id='articles',
		instruction='点击侧边栏或导航中的“文章”。',
		expected_result='显示文章列表。',
		expectation_source=ExpectationSource.EXPLICIT,
		source_evidence=['1. 点击文章；预期结果：显示文章列表。'],
	)
	evidence = _strict_evidence(
		action_status=ActionCompletionStatus.COMPLETED,
		target_matched=None,
		evidence_quality=EvidenceQuality.WEAK,
	)
	evidence.after = BrowserEvidenceSnapshot(
		url='https://demo.halocms.site/console/posts',
		title='Halo 演示站点',
		dom_summary='文章 分类 标签 回收站 新建',
		artifacts=[EvidenceArtifact(evidence_id='ev_dom', kind=EvidenceKind.DOM, summary='文章 分类 标签 回收站 新建')],
	)
	llm = _mock_llm(
		StepJudgement(
			action_status=ActionCompletionStatus.COMPLETED,
			expectation_status=ExpectationStatus.MET,
			status=QAStepStatus.PASSED,
			failure_origin=FailureOrigin.NONE,
			reasoning='The after-state is the posts page and shows article list controls.',
			actual_result='The article list page is visible.',
			evidence_ids=['ev_dom'],
		)
	)

	judgement = await judge_test_step(llm=llm, step=step, evidence=evidence)

	assert judgement.status == QAStepStatus.PASSED
	assert judgement.failure_origin == FailureOrigin.NONE
	assert llm.ainvoke.await_count == 1
