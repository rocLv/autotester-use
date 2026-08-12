import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import browser_use.agent.service as agent_service
from browser_use import Agent
from browser_use.agent.views import ActionResult, AgentHistory, BrowserStateHistory
from browser_use.browser import BrowserProfile
from browser_use.browser.events import BrowserStartEvent
from browser_use.browser.session import ResilientEventBus
from browser_use.dom.views import DOMInteractedElement, DOMRect, NodeType
from browser_use.llm.exceptions import ModelProviderError
from browser_use.qa.compiler import QATaskCompiler, WebUITestCaseDraft
from browser_use.qa.views import (
	ActionCompletionStatus,
	BrowserEvidenceSnapshot,
	ExpectationSource,
	ExpectationStatus,
	FailureOrigin,
	PreconditionMode,
	QAPrecondition,
	QARunStatus,
	QAStepResult,
	QAStepStatus,
	ReplayAssertion,
	ReplayAssertionKind,
	StepEvidence,
	StepJudgement,
	StepOperationKind,
	WebUITestCase,
	WebUITestStep,
)
from tests.ci.conftest import create_mock_llm


def _case(step_count: int = 2) -> WebUITestCase:
	return WebUITestCase(
		root_url='https://example.com/app',
		registrable_domain='example.com',
		steps=[
			WebUITestStep(
				step_id=f'step-{index + 1}',
				instruction=f'Do business action {index + 1}',
				expected_result=f'Observable result {index + 1} is visible',
				expectation_source=ExpectationSource.EXPLICIT,
				source_evidence=[f'Expected observable result {index + 1}'],
			)
			for index in range(step_count)
		],
	)


def _evidence() -> StepEvidence:
	return StepEvidence(
		before=BrowserEvidenceSnapshot(url='https://example.com/app', dom_summary='Before'),
		after=BrowserEvidenceSnapshot(url='https://example.com/app', dom_summary='After'),
	)


def _passed_replay_baseline(step: WebUITestStep) -> QAStepResult:
	return QAStepResult(
		step=step,
		status=QAStepStatus.PASSED,
		judgement=StepJudgement(
			action_status=ActionCompletionStatus.COMPLETED,
			expectation_status=ExpectationStatus.MET,
			status=QAStepStatus.PASSED,
			failure_origin=FailureOrigin.NONE,
			reasoning='The expected marker was visible in the reliable first run.',
			actual_result='Expected marker visible.',
			replay_assertions=[ReplayAssertion(kind=ReplayAssertionKind.DOM_CONTAINS, value='Expected marker')],
		),
	)


def _boundary_history() -> AgentHistory:
	return AgentHistory(
		model_output=None,
		result=[ActionResult(is_done=True, metadata={'qa_finish_test_step': {'actual_result': 'After'}})],
		state=BrowserStateHistory(url='https://example.com/app', title='App', tabs=[], interacted_element=[]),
		metadata=None,
	)


def _agent(step_count: int = 2) -> Agent:
	agent = Agent(task='Test https://example.com/app', llm=create_mock_llm())
	agent._qa_test_case = _case(step_count)
	agent._qa_before_snapshot = _evidence().before
	agent._build_qa_evidence = AsyncMock(return_value=_evidence())
	agent.history.add_item(_boundary_history())
	return agent


def test_custom_tool_target_proof_completes_input_action_receipt():
	agent = _agent(step_count=1)
	step = agent._qa_test_case.steps[0].model_copy(update={'operation_kind': StepOperationKind.INPUT})  # type: ignore[union-attr]
	receipt, artifact = agent._build_action_receipt(
		step=step,
		before=BrowserEvidenceSnapshot(url='https://example.com/app', dom_summary='Empty form'),
		after=BrowserEvidenceSnapshot(url='https://example.com/app', dom_summary='Completed form'),
		action_results=[
			ActionResult(
				extracted_content='Form fields verified',
				metadata={
					'qa_target_proof': {
						'target_name': 'registration form fields',
						'target_matched': True,
						'verification': {'phone_valid': True, 'password_valid': True},
					}
				},
			).model_dump(exclude_none=True, mode='json')
		],
		action_names=['fill_registration_form'],
		selected_element=None,
		input_values=[],
		side_effect_uncertain=False,
	)

	assert receipt.status == ActionCompletionStatus.COMPLETED
	assert receipt.target_matched is True
	assert artifact.metadata['tool_target_proofs'][0]['target_name'] == 'registration form fields'


def test_failed_custom_tool_cannot_use_target_proof_to_complete_receipt():
	agent = _agent(step_count=1)
	step = agent._qa_test_case.steps[0].model_copy(update={'operation_kind': StepOperationKind.INPUT})  # type: ignore[union-attr]
	receipt, _ = agent._build_action_receipt(
		step=step,
		before=BrowserEvidenceSnapshot(url='https://example.com/app', dom_summary='Empty form'),
		after=BrowserEvidenceSnapshot(url='https://example.com/app', dom_summary='Empty form'),
		action_results=[
			ActionResult(
				error='Input dispatch failed',
				metadata={
					'qa_target_proof': {
						'target_name': 'registration form fields',
						'target_matched': True,
						'verification': {'phone_valid': True},
					}
				},
			).model_dump(exclude_none=True, mode='json')
		],
		action_names=['fill_registration_form'],
		selected_element=None,
		input_values=[],
		side_effect_uncertain=False,
	)

	assert receipt.status == ActionCompletionStatus.NOT_COMPLETED
	assert receipt.tool_succeeded is False
	assert receipt.target_matched is None


@pytest.mark.asyncio
async def test_pass_advances_to_next_business_step(monkeypatch):
	agent = _agent()
	monkeypatch.setattr(
		agent_service,
		'judge_test_step',
		AsyncMock(
			return_value=StepJudgement(
				action_status=ActionCompletionStatus.COMPLETED,
				expectation_status=ExpectationStatus.MET,
				status=QAStepStatus.PASSED,
				failure_origin=FailureOrigin.NONE,
				reasoning='Expected state is visible.',
				actual_result='Expected state is visible.',
			)
		),
	)
	assert await agent._handle_qa_step_boundary() is False
	assert agent._qa_current_step_index == 1
	assert agent.history.history[-1].result[-1].is_done is False


@pytest.mark.asyncio
async def test_sut_failure_stops_and_marks_remaining_steps_not_run(monkeypatch):
	agent = _agent()
	judge = AsyncMock(
		return_value=StepJudgement(
			action_status=ActionCompletionStatus.COMPLETED,
			expectation_status=ExpectationStatus.NOT_MET,
			status=QAStepStatus.SUT_FAILED,
			failure_origin=FailureOrigin.SUT,
			reasoning='Action completed but expected state is absent.',
			actual_result='Expected state is absent.',
		)
	)
	monkeypatch.setattr(
		agent_service,
		'judge_test_step',
		judge,
	)
	assert await agent._handle_qa_step_boundary() is True
	assert agent.history.qa_result is not None
	assert agent.history.qa_result.status == QARunStatus.SUT_FAILED
	assert [result.status for result in agent.history.qa_result.step_results] == [
		QAStepStatus.SUT_FAILED,
		QAStepStatus.NOT_RUN,
	]
	assert agent.history.is_successful() is False
	assert judge.await_count == 2
	assert agent.history.qa_result.step_results[0].review is not None
	assert agent.history.qa_result.step_results[0].review.agreed is True


@pytest.mark.asyncio
async def test_sut_review_disagreement_becomes_inconclusive(monkeypatch):
	agent = _agent(step_count=1)
	primary = StepJudgement(
		action_status=ActionCompletionStatus.COMPLETED,
		expectation_status=ExpectationStatus.NOT_MET,
		status=QAStepStatus.SUT_FAILED,
		failure_origin=FailureOrigin.SUT,
		reasoning='Expected state absent.',
		actual_result='Expected state absent.',
	)
	secondary = StepJudgement(
		action_status=ActionCompletionStatus.COMPLETED,
		expectation_status=ExpectationStatus.MET,
		status=QAStepStatus.PASSED,
		failure_origin=FailureOrigin.NONE,
		reasoning='Expected state visible.',
		actual_result='Expected state visible.',
	)
	monkeypatch.setattr(agent_service, 'judge_test_step', AsyncMock(side_effect=[primary, secondary]))

	assert await agent._handle_qa_step_boundary() is True
	assert agent.history.qa_result is not None
	assert agent.history.qa_result.status == QARunStatus.INCONCLUSIVE
	assert agent.history.is_successful() is None


@pytest.mark.asyncio
async def test_agent_failure_retries_at_most_three_times(monkeypatch):
	agent = _agent(step_count=1)
	judgement = StepJudgement(
		action_status=ActionCompletionStatus.NOT_COMPLETED,
		expectation_status=ExpectationStatus.NOT_OBSERVABLE,
		status=QAStepStatus.AGENT_FAILED,
		failure_origin=FailureOrigin.AGENT,
		reasoning='The click did not occur and no state changed.',
		actual_result='No state change.',
		retry_safe=True,
	)
	monkeypatch.setattr(agent_service, 'judge_test_step', AsyncMock(return_value=judgement))

	for retry in range(3):
		assert await agent._handle_qa_step_boundary() is False
		assert agent._qa_step_retry_count == retry + 1
		agent.history.add_item(_boundary_history())

	assert await agent._handle_qa_step_boundary() is True
	assert agent.history.qa_result is not None
	assert agent.history.qa_result.status == QARunStatus.AGENT_FAILED
	assert agent.history.qa_result.step_results[0].retry_count == 3
	assert agent.history.is_successful() is None


@pytest.mark.asyncio
async def test_uncertain_side_effect_is_never_replayed(monkeypatch):
	agent = _agent(step_count=1)
	agent._build_qa_evidence = AsyncMock(
		return_value=StepEvidence(
			before=_evidence().before,
			after=_evidence().after,
			side_effect_uncertain=True,
		)
	)
	monkeypatch.setattr(
		agent_service,
		'judge_test_step',
		AsyncMock(
			return_value=StepJudgement(
				action_status=ActionCompletionStatus.NOT_COMPLETED,
				expectation_status=ExpectationStatus.NOT_OBSERVABLE,
				status=QAStepStatus.AGENT_FAILED,
				failure_origin=FailureOrigin.AGENT,
				reasoning='The submit may have reached the server.',
				actual_result='Submission state is uncertain.',
				retry_safe=True,
			)
		),
	)
	assert await agent._handle_qa_step_boundary() is True
	assert agent.history.qa_result is not None
	assert agent.history.qa_result.status == QARunStatus.AGENT_FAILED
	assert agent.history.qa_result.step_results[0].retry_count == 0


@pytest.mark.asyncio
async def test_cleanup_failure_is_recorded_without_overwriting_business_verdict(monkeypatch):
	agent = _agent(step_count=1)
	assert agent._qa_test_case is not None
	agent._qa_test_case.cleanup_steps = [
		WebUITestStep(
			step_id='cleanup',
			instruction='Remove the run-scoped test record',
			expected_result='The test record is absent',
			expectation_source=ExpectationSource.EXPLICIT,
			source_evidence=['The test record is absent'],
		)
	]
	passed = StepJudgement(
		action_status=ActionCompletionStatus.COMPLETED,
		expectation_status=ExpectationStatus.MET,
		status=QAStepStatus.PASSED,
		failure_origin=FailureOrigin.NONE,
		reasoning='Business expectation met.',
		actual_result='Business expectation met.',
		confidence=1,
	)
	cleanup_failed = StepJudgement(
		action_status=ActionCompletionStatus.NOT_COMPLETED,
		expectation_status=ExpectationStatus.NOT_OBSERVABLE,
		status=QAStepStatus.AGENT_FAILED,
		failure_origin=FailureOrigin.AGENT,
		reasoning='Cleanup target could not be selected.',
		actual_result='Cleanup did not complete.',
	)
	monkeypatch.setattr(agent_service, 'judge_test_step', AsyncMock(side_effect=[passed, cleanup_failed]))

	assert await agent._handle_qa_step_boundary() is False
	assert agent._qa_running_cleanup is True
	agent.history.add_item(_boundary_history())
	assert await agent._handle_qa_step_boundary() is True

	assert agent.history.qa_result is not None
	assert agent.history.qa_result.status == QARunStatus.PASSED
	assert agent.history.qa_result.cleanup_results[0].status == QAStepStatus.AGENT_FAILED
	assert 'Cleanup cleanup ended as AGENT_FAILED' in agent.history.qa_result.warnings[0]


@pytest.mark.asyncio
async def test_explicit_case_performs_only_one_runner_managed_root_navigation(monkeypatch):
	agent = _agent(step_count=1)
	assert agent._qa_test_case is not None
	draft = WebUITestCaseDraft.model_validate(
		{
			'steps': [
				{
					'step_id': 'step-1',
					'instruction': 'Inspect the result',
					'expected_result': 'Observable result 1 is visible',
					'source_evidence': ['Observable result 1 is visible'],
				}
			]
		}
	)
	navigate = AsyncMock()
	monkeypatch.setattr(type(agent.browser_session), 'navigate_to', navigate)
	agent._wait_for_qa_page_stability = AsyncMock()
	agent._start_qa_execution = AsyncMock()
	monkeypatch.setattr(
		QATaskCompiler,
		'complete_with_discovery',
		AsyncMock(return_value=agent._qa_test_case),
	)

	await agent._prepare_qa_case(draft)

	navigate.assert_awaited_once_with('https://example.com/app', new_tab=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
	('step_status', 'failure_origin', 'run_status'),
	[
		(QAStepStatus.BLOCKED, FailureOrigin.ENVIRONMENT, QARunStatus.BLOCKED),
		(QAStepStatus.INCONCLUSIVE, FailureOrigin.UNKNOWN, QARunStatus.INCONCLUSIVE),
	],
)
async def test_non_product_failures_stop_without_a_false_sut_verdict(monkeypatch, step_status, failure_origin, run_status):
	agent = _agent()
	monkeypatch.setattr(
		agent_service,
		'judge_test_step',
		AsyncMock(
			return_value=StepJudgement(
				action_status=ActionCompletionStatus.UNCERTAIN,
				expectation_status=ExpectationStatus.NOT_OBSERVABLE,
				status=step_status,
				failure_origin=failure_origin,
				reasoning='The expected state cannot be reliably observed.',
				actual_result='No reliable product verdict is available.',
			)
		),
	)
	assert await agent._handle_qa_step_boundary() is True
	assert agent.history.qa_result is not None
	assert agent.history.qa_result.status == run_status
	assert agent.history.is_successful() is None
	assert agent.history.qa_result.step_results[1].status == QAStepStatus.NOT_RUN


@pytest.mark.asyncio
async def test_missing_precondition_secret_blocks_before_business_execution():
	agent = _agent(step_count=1)
	assert agent._qa_test_case is not None
	agent._qa_test_case.preconditions = [
		QAPrecondition(
			precondition_id='login',
			description='Log in with HALO_PASSWORD',
			mode=PreconditionMode.ENSURE,
			sensitive_refs=['HALO_PASSWORD'],
		)
	]
	agent._capture_qa_snapshot = AsyncMock(return_value=_evidence().before)

	await agent._start_qa_execution()

	assert agent.history.qa_result is not None
	assert agent.history.qa_result.status == QARunStatus.BLOCKED
	assert all(result.status == QAStepStatus.NOT_RUN for result in agent.history.qa_result.step_results)


@pytest.mark.asyncio
async def test_unsatisfied_precondition_is_blocked_not_sut_failed(monkeypatch):
	agent = _agent(step_count=1)
	assert agent._qa_test_case is not None
	precondition = QAPrecondition(precondition_id='role', description='The admin role is active')
	agent._qa_test_case.preconditions = [precondition]
	agent._qa_running_preconditions = True
	agent._qa_precondition_step = WebUITestStep(
		step_id='__preconditions__',
		instruction=precondition.description,
		expected_result='Every required precondition is satisfied.',
		expectation_source=ExpectationSource.EXPLICIT,
		source_evidence=[precondition.description],
	)
	monkeypatch.setattr(
		agent_service,
		'judge_test_step',
		AsyncMock(
			return_value=StepJudgement(
				action_status=ActionCompletionStatus.COMPLETED,
				expectation_status=ExpectationStatus.NOT_MET,
				status=QAStepStatus.SUT_FAILED,
				failure_origin=FailureOrigin.SUT,
				reasoning='Required role is absent.',
				actual_result='Required role is absent.',
			)
		),
	)

	assert await agent._handle_qa_step_boundary() is True
	assert agent.history.qa_result is not None
	assert agent.history.qa_result.status == QARunStatus.BLOCKED


def test_model_free_replay_attributes_missing_expectation_to_sut_only_after_action_completed():
	agent = _agent(step_count=1)
	assert agent._qa_test_case is not None
	step = agent._qa_test_case.steps[0]
	baseline = _passed_replay_baseline(step)
	evidence = StepEvidence(
		before=BrowserEvidenceSnapshot(url='https://example.com/app', dom_summary='Before'),
		after=BrowserEvidenceSnapshot(url='https://example.com/app', dom_summary='Unexpected page'),
	)

	sut_judgement = agent._judge_qa_replay_step(
		step=step,
		baseline=baseline,
		evidence=evidence,
		last_interaction_completed=True,
		had_interactions=True,
	)
	agent_judgement = agent._judge_qa_replay_step(
		step=step,
		baseline=baseline,
		evidence=evidence,
		last_interaction_completed=False,
		had_interactions=True,
	)

	assert sut_judgement.status == QAStepStatus.SUT_FAILED
	assert sut_judgement.failure_origin == FailureOrigin.SUT
	assert agent_judgement.status == QAStepStatus.AGENT_FAILED
	assert agent_judgement.failure_origin == FailureOrigin.AGENT


def test_agent_rejects_legacy_completion_controls():
	llm = create_mock_llm()
	valid_agent = Agent(task='Test https://example.com', llm=llm)
	action_schema = valid_agent.ActionModel.model_json_schema()
	assert 'finish_test_step' in str(action_schema)
	assert '"done"' not in json.dumps(action_schema)
	with pytest.raises(ValueError, match='requires use_judge=True'):
		Agent(task='Test https://example.com', llm=llm, use_judge=False)
	with pytest.raises(ValueError, match='initial_actions are not supported'):
		Agent(
			task='Test https://example.com',
			llm=llm,
			initial_actions=[{'navigate': {'url': 'https://example.com'}}],
		)
	with pytest.raises(ValueError, match='between 0 and 3'):
		Agent(task='Test https://example.com', llm=llm, max_agent_retries_per_step=4)


def test_persistence_applies_defense_in_depth_sensitive_data_redaction():
	agent = Agent(
		task='Test https://example.com/app',
		llm=create_mock_llm(),
		sensitive_data={'SECRET_TOKEN': 'top-secret-value'},
	)
	result = agent_service.QARunResult(
		status=QARunStatus.AGENT_FAILED,
		failure_origin=FailureOrigin.AGENT,
		summary='Diagnostic accidentally included top-secret-value',
	)

	redacted = agent._redacted_qa_result_for_persistence(result)

	assert 'top-secret-value' not in redacted.model_dump_json()


def test_current_step_prompt_includes_case_and_step_preconditions():
	agent = _agent(step_count=1)
	assert agent._qa_test_case is not None
	agent._qa_test_case = agent._qa_test_case.model_copy(
		update={
			'preconditions': [
				QAPrecondition(
					precondition_id='login',
					description='Log in as demo',
					mode=PreconditionMode.ENSURE,
				)
			]
		}
	)
	agent._qa_test_case.steps[0].preconditions.append(
		QAPrecondition(
			precondition_id='dashboard_visible',
			description='The dashboard is visible',
			mode=PreconditionMode.VERIFY,
		)
	)

	prompt = agent._qa_step_prompt(agent._qa_test_case.steps[0])

	assert "Test case preconditions: ['Log in as demo']" in prompt
	assert "Step preconditions: ['The dashboard is visible']" in prompt


def test_structured_table_log_redacts_registered_sensitive_values():
	agent = _agent(step_count=1)
	assert agent._qa_test_case is not None
	agent._qa_test_case.preconditions = [
		QAPrecondition(
			precondition_id='login',
			description='Log in with password top-secret',
			mode=PreconditionMode.ENSURE,
		)
	]
	sensitive_data: dict[str, str | dict[str, str]] = {'QA_PASSWORD': 'top-secret'}
	agent.sensitive_data = sensitive_data

	with patch.object(agent.logger, 'info') as log_info:
		agent._log_qa_test_case_table('Structured test')

	message = log_info.call_args.args[0]
	assert 'top-secret' not in message
	assert '<secret>QA_PASSWORD</secret>' in message


def test_agent_detects_empty_intersection_with_caller_navigation_policy():
	agent = Agent(
		task='Open https://example.com/app and verify the title is visible.',
		llm=create_mock_llm(),
		browser_profile=BrowserProfile(allowed_domains=['internal.test']),
	)
	assert agent._qa_scope_error is not None
	assert 'conflicts' in agent._qa_scope_error


def test_add_new_task_recompiles_in_scope_and_rejects_cross_domain_followups():
	agent = _agent()
	agent.add_new_task('Open https://admin.example.com/settings and verify the title is visible.')
	assert agent._qa_test_case is None
	assert agent._qa_scope is not None
	assert agent._qa_scope.root_host == 'admin.example.com'
	with pytest.raises(ValueError, match='cannot change registrable domain'):
		agent.add_new_task('Open https://other.com and verify the title is visible.')


def test_same_task_followup_reuses_compiled_case_by_default():
	agent = _agent()
	assert agent._qa_test_case is not None
	cached_case = agent._qa_test_case
	agent._qa_compiled_task = agent._qa_original_task

	agent.add_new_task(agent._qa_original_task)

	assert agent._qa_test_case is cached_case
	assert agent._qa_test_case_draft is None
	assert agent.history.qa_result is None
	assert agent.state.n_steps == 1


@pytest.mark.asyncio
async def test_completed_rerun_does_not_send_the_previous_executor_transcript_to_the_model():
	agent = _agent()
	agent._qa_compiled_task = agent._qa_original_task
	agent._finalize_qa_result(
		status=QARunStatus.PASSED,
		failure_origin=FailureOrigin.NONE,
		summary='First run passed.',
	)
	agent.state.message_manager_state.compacted_memory = 'previous executor transcript'
	await agent.eventbus.stop(clear=True)

	agent.add_new_task(agent._qa_original_task)

	assert agent.state.message_manager_state.compacted_memory is None
	assert 'previous executor transcript' not in agent.message_manager.agent_history_description
	assert 'Reusing the validated QA specification' in agent.message_manager.agent_history_description


def test_rerun_restores_browser_lifecycle_handlers_cleared_by_close():
	agent = _agent()
	agent.browser_session.event_bus = ResilientEventBus()
	assert agent.browser_session.event_bus.handlers.get(BrowserStartEvent.__name__, []) == []

	agent._ensure_browser_session_can_restart()

	assert agent.browser_session.event_bus.handlers.get(BrowserStartEvent.__name__, [])


def test_strict_replay_caps_saved_fixed_waits():
	agent = _agent(step_count=1)
	AgentOutput = agent.AgentOutput
	history_item = AgentHistory(
		model_output=AgentOutput(
			evaluation_previous_goal=None,
			memory='Wait for the authenticated dashboard',
			next_goal=None,
			action=[{'wait': {'seconds': 5}}],  # type: ignore[arg-type]
		),
		result=[ActionResult(extracted_content='Waited')],
		state=BrowserStateHistory(url='https://example.com/login', title='Login', tabs=[], interacted_element=[None]),
		metadata=None,
	)

	prepared, action_names, error = agent._prepare_qa_replay_history_item(history_item)

	assert error is None
	assert action_names == ['wait']
	assert prepared is not None and prepared.model_output is not None
	assert prepared.model_output.action[0].model_dump(exclude_unset=True)['wait']['seconds'] == 1
	assert history_item.model_output is not None
	assert history_item.model_output.action[0].model_dump(exclude_unset=True)['wait']['seconds'] == 5


def test_strict_replay_skips_obsolete_authentication_batch_after_login_restore():
	agent = _agent(step_count=1)
	agent._qa_login_storage_state = {'cookies': [{'name': 'session', 'value': 'token'}], 'origins': []}
	AgentOutput = agent.AgentOutput
	login_button = DOMInteractedElement(
		node_id=1,
		backend_node_id=1,
		frame_id=None,
		node_type=NodeType.ELEMENT_NODE,
		node_value='',
		node_name='BUTTON',
		attributes={'id': 'login'},
		x_path='html/body/button',
		element_hash=123,
		stable_hash=123,
		bounds=DOMRect(x=0, y=0, width=100, height=40),
	)
	history_item = AgentHistory(
		model_output=AgentOutput(
			evaluation_previous_goal=None,
			memory='Submit login',
			next_goal=None,
			action=[{'click': {'index': 1}}],  # type: ignore[arg-type]
		),
		result=[ActionResult(extracted_content='Logged in')],
		state=BrowserStateHistory(url='https://example.com/login', title='Login', tabs=[], interacted_element=[login_button]),
		metadata=None,
	)
	current_state = SimpleNamespace(
		url='https://example.com/app',
		dom_state=SimpleNamespace(selector_map={}),
	)

	assert agent._qa_replay_authentication_batch_is_obsolete(history_item, current_state) is True  # type: ignore[arg-type]


def test_same_task_followup_can_disable_compiled_case_reuse():
	agent = Agent(
		task='Test https://example.com/app',
		llm=create_mock_llm(),
		qa_test_case=_case(),
		reuse_compiled_test_case=False,
	)

	agent.add_new_task(agent._qa_original_task)

	assert agent._qa_test_case is None


def test_qa_step_limit_trims_compiled_case_inside_agent_core():
	agent = Agent(
		task='Test https://example.com/app',
		llm=create_mock_llm(),
		qa_test_case=_case(step_count=3),
		qa_step_limit=1,
	)

	agent._apply_qa_execution_limits()

	assert agent._qa_test_case is not None
	assert [step.step_id for step in agent._qa_test_case.steps] == ['step-1']
	assert agent._qa_warnings == ['QA execution limited to first 1 of 3 compiled business steps.']


def test_cached_login_state_is_limited_to_qa_navigation_scope():
	agent = _agent()
	assert agent.settings.reuse_login_state is True
	filtered = agent._filter_qa_login_storage_state(
		{
			'cookies': [
				{'name': 'root', 'value': '1', 'domain': '.example.com', 'path': '/'},
				{'name': 'subdomain', 'value': '2', 'domain': 'admin.example.com', 'path': '/'},
				{'name': 'external', 'value': '3', 'domain': 'evil.test', 'path': '/'},
			],
			'origins': [
				{'origin': 'https://admin.example.com', 'localStorage': []},
				{'origin': 'https://evil.test', 'localStorage': []},
			],
		}
	)

	assert [cookie['name'] for cookie in filtered['cookies']] == ['root', 'subdomain']
	assert [origin['origin'] for origin in filtered['origins']] == ['https://admin.example.com']


@pytest.mark.asyncio
async def test_cached_login_state_is_restored_before_rerun_navigation(monkeypatch):
	agent = _agent()
	agent._qa_login_storage_state = {
		'cookies': [{'name': 'session', 'value': 'token', 'domain': 'example.com', 'path': '/', 'expires': -1}],
		'origins': [
			{
				'origin': 'https://example.com',
				'localStorage': [{'name': 'access_token', 'value': 'secret'}],
				'sessionStorage': [{'name': 'tenant', 'value': 'qa'}],
			}
		],
	}
	set_cookies = AsyncMock()
	add_init_script = AsyncMock()
	monkeypatch.setattr(agent.browser_session, '_cdp_set_cookies', set_cookies)
	monkeypatch.setattr(agent.browser_session, '_cdp_add_init_script', add_init_script)

	await agent._restore_cached_qa_login_state()

	set_cookies.assert_awaited_once()
	assert set_cookies.await_args is not None
	assert 'expires' not in set_cookies.await_args.args[0][0]
	add_init_script.assert_awaited_once()
	assert add_init_script.await_args is not None
	assert 'access_token' in add_init_script.await_args.args[0]


@pytest.mark.asyncio
async def test_invalid_spec_does_not_start_browser(monkeypatch):
	done_callback = AsyncMock()
	agent = Agent(
		task='Verify the login page without a URL',
		llm=create_mock_llm(),
		register_done_callback=done_callback,
	)
	start = AsyncMock()
	monkeypatch.setattr(type(agent.browser_session), 'start', start)
	monkeypatch.setattr(agent, 'close', AsyncMock())
	history = await agent.run(max_steps=1)
	assert history.qa_result is not None
	assert history.qa_result.status == QARunStatus.INVALID_SPEC
	start.assert_not_awaited()
	done_callback.assert_awaited_once_with(history)
	closed_eventbus = agent.eventbus
	agent.add_new_task('Open https://example.com and verify the title is visible.')
	assert agent.eventbus is not closed_eventbus


@pytest.mark.asyncio
async def test_specification_model_failure_is_agent_failed_without_starting_browser(monkeypatch):
	agent = Agent(
		task='Open https://example.com and verify the title is visible.',
		llm=create_mock_llm(),
	)
	start = AsyncMock()
	monkeypatch.setattr(type(agent.browser_session), 'start', start)
	monkeypatch.setattr(
		agent_service.QATaskCompiler,
		'extract_requirements',
		AsyncMock(side_effect=ModelProviderError('judge model unavailable')),
	)
	monkeypatch.setattr(agent, 'close', AsyncMock())

	history = await agent.run(max_steps=1)

	assert history.qa_result is not None
	assert history.qa_result.status == QARunStatus.AGENT_FAILED
	assert history.qa_result.failure_origin == FailureOrigin.AGENT
	assert history.is_successful() is None
	start.assert_not_awaited()


def test_runtime_browser_failure_is_blocked_but_plain_framework_failure_is_agent_failed():
	assert Agent._qa_runtime_failure_status(RuntimeError('CDP websocket closed'), phase='execution') == (
		QARunStatus.BLOCKED,
		FailureOrigin.ENVIRONMENT,
	)
	assert Agent._qa_runtime_failure_status(RuntimeError('executor could not choose an action'), phase='execution') == (
		QARunStatus.AGENT_FAILED,
		FailureOrigin.AGENT,
	)
