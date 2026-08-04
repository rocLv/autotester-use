from browser_use.agent.views import AgentHistoryList, AgentOutput
from browser_use.qa.views import (
	ExpectationSource,
	FailureOrigin,
	QARunResult,
	QARunStatus,
	QAStepResult,
	QAStepStatus,
	SideEffectLevel,
	WebUITestCase,
	WebUITestStep,
)


def _test_case() -> WebUITestCase:
	return WebUITestCase(
		root_url='https://example.com/app',
		registrable_domain='example.com',
		steps=[
			WebUITestStep(
				step_id='step-1',
				instruction='Submit an empty required field',
				expected_result='A required-field validation message is visible',
				expectation_source=ExpectationSource.EXPLICIT,
				source_evidence=['Expected: required-field validation message'],
			)
		],
	)


def test_history_success_only_represents_reliable_product_verdicts():
	case = _test_case()
	history = AgentHistoryList(history=[])
	history.qa_result = QARunResult(
		status=QARunStatus.PASSED,
		failure_origin=FailureOrigin.NONE,
		test_case=case,
		step_results=[QAStepResult(step=case.steps[0], status=QAStepStatus.PASSED)],
		summary='passed',
	)
	assert history.is_done() is True
	assert history.is_successful() is True

	history.qa_result = QARunResult(
		status=QARunStatus.SUT_FAILED,
		failure_origin=FailureOrigin.SUT,
		test_case=case,
		summary='failed',
	)
	assert history.is_successful() is False

	history.qa_result = QARunResult(
		status=QARunStatus.AGENT_FAILED,
		failure_origin=FailureOrigin.AGENT,
		test_case=case,
		summary='agent failed',
	)
	assert history.is_successful() is None


def test_history_serializes_qa_result():
	history = AgentHistoryList(
		history=[],
		qa_result=QARunResult(
			status=QARunStatus.INVALID_SPEC,
			failure_origin=None,
			summary='missing URL',
			validation_errors=['missing URL'],
		),
	)
	dumped = history.model_dump()
	assert dumped['qa_result']['status'] == 'INVALID_SPEC'
	loaded = AgentHistoryList.model_validate(dumped)
	assert loaded.qa_result is not None
	assert loaded.qa_result.failure_origin is None


def test_test_case_renders_strongly_typed_document_as_markdown_tables():
	case = _test_case().model_copy(update={'preconditions': ['Log in as demo', 'Open Posts']})
	case.steps[0].instruction = 'Click Save | Publish'

	markdown = case.to_markdown_table()

	assert '| 用例字段 | 值 |' in markdown
	assert '| 起始 URL | https://example.com/app |' in markdown
	assert '| 全局前置条件 | VERIFY precondition_1: Log in as demo<br>VERIFY precondition_2: Open Posts |' in markdown
	assert '| 步骤 ID | 步骤前置条件 | 操作类型 | 副作用级别 | 幂等标识 | 操作 | 预期结果 | 预期来源 | 需求引用 |' in markdown
	assert 'Click Save \\| Publish' in markdown


def test_side_effecting_steps_receive_a_run_scoped_idempotency_key():
	step = (
		_test_case()
		.steps[0]
		.model_copy(
			update={'side_effect_level': SideEffectLevel.IRREVERSIBLE},
		)
	)
	step = WebUITestStep.model_validate(step.model_dump())

	assert step.idempotency_key == 'autotester:step-1:${run_id}'


def test_v1_history_is_migrated_for_display_but_not_trusted_for_replay():
	case = _test_case()
	payload = {
		'history': [],
		'qa_result': {
			'status': 'PASSED',
			'failure_origin': 'none',
			'test_case': {
				**case.model_dump(mode='json'),
				'preconditions': ['Log in as demo'],
			},
			'step_results': [],
			'summary': 'legacy pass',
		},
	}

	loaded = AgentHistoryList.load_from_dict(payload, AgentOutput)

	assert loaded.qa_result is not None
	assert loaded.qa_result.schema_version == 2
	assert loaded.qa_result.legacy_imported is True
	assert loaded.qa_result.has_reliable_verdict is False
	assert loaded.qa_result.test_case is not None
	assert loaded.qa_result.test_case.preconditions[0].description == 'Log in as demo'
