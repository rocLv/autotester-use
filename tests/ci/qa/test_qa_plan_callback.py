from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

import browser_use.agent.service as agent_service
from browser_use import Agent
from browser_use.llm import BaseChatModel
from browser_use.llm.views import ChatInvokeCompletion
from browser_use.qa.compiler import WebUITestCaseDraft
from browser_use.qa.views import (
	ActionCompletionStatus,
	ActionReceipt,
	BrowserEvidenceSnapshot,
	EvidenceArtifact,
	EvidenceKind,
	ExpectationSource,
	ExpectationStatus,
	FailureOrigin,
	QAPlanSnapshot,
	QAPrecondition,
	QARunResult,
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

_SIMPLE_TASK = (
	'Open https://example.com/app. Step 1: Perform action 1. Expected result 1. Step 2: Perform action 2. Expected result 2.'
)


def _article_task() -> str:
	return """测试用例名称：发布文章
所属模块：文章
前置条件：
- 后台地址：https://demo.halocms.site/console
- 用户名：demo
- 密码：P@ssw0rd123..
- 进入管理登录页：https://demo.halocms.site/console/dashboard

测试步骤：
1. 点击文章；预期结果：显示文章列表
2. 点击右上角 新建；预期结果：显示文章编辑页
3. 输入文章标题、正文内容，点击发布；预期结果：弹出文章设置弹窗
4. 弹窗内点击 发布；预期结果：
   1. 管理系统列表新增一条文章记录
   2. Halo 首页新增一篇文章"""


def _draft_with_steps(*, missing_expected: bool = False) -> WebUITestCaseDraft:
	steps = []
	for index in range(2):
		expected_result = None if missing_expected and index == 1 else f'Expected result {index + 1}'
		steps.append(
			{
				'step_id': f'step_{index + 1}',
				'instruction': f'Perform action {index + 1}',
				'expected_result': expected_result,
				'source_evidence': [] if expected_result is None else [expected_result],
				'operation_kind': 'click',
			}
		)
	return WebUITestCaseDraft.model_validate(
		{
			'preconditions': ['Open https://example.com/app', 'Password: P@ssw0rd123..'],
			'steps': steps,
		}
	)


def _article_draft() -> WebUITestCaseDraft:
	return WebUITestCaseDraft.model_validate(
		{
			'preconditions': [
				'后台地址：https://demo.halocms.site/console',
				'用户名：demo',
				'密码：P@ssw0rd123..',
				'进入管理登录页：https://demo.halocms.site/console/dashboard',
			],
			'steps': [
				{
					'step_id': 'step_1',
					'instruction': '点击文章',
					'expected_result': '显示文章列表',
					'source_evidence': ['显示文章列表'],
					'operation_kind': 'click',
				},
				{
					'step_id': 'step_2',
					'instruction': '点击右上角 新建',
					'expected_result': '显示文章编辑页',
					'source_evidence': ['显示文章编辑页'],
					'operation_kind': 'click',
				},
				{
					'step_id': 'step_3',
					'instruction': '输入文章标题、正文内容，点击发布',
					'expected_result': '弹出文章设置弹窗',
					'source_evidence': ['弹出文章设置弹窗'],
					'operation_kind': 'submit',
					'side_effect_level': 'irreversible',
				},
				{
					'step_id': 'step_4',
					'instruction': '弹窗内点击 发布',
					'expected_result': '管理系统列表新增一条文章记录\n   2. Halo 首页新增一篇文章',
					'source_evidence': ['管理系统列表新增一条文章记录\n   2. Halo 首页新增一篇文章'],
					'operation_kind': 'submit',
					'side_effect_level': 'irreversible',
				},
			],
		}
	)


def _case_from_draft(draft: WebUITestCaseDraft) -> WebUITestCase:
	return WebUITestCase(
		root_url='https://example.com/app',
		registrable_domain='example.com',
		preconditions=[
			QAPrecondition(precondition_id=f'precondition_{index}', description=value)
			for index, value in enumerate(draft.preconditions, start=1)
		],
		steps=[
			WebUITestStep(
				step_id=step.step_id,
				instruction=step.instruction,
				expected_result=step.expected_result or f'Discovered result for {step.step_id}',
				expectation_source=ExpectationSource.EXPLICIT if step.expected_result else ExpectationSource.HEURISTIC,
				operation_kind=step.operation_kind,
				side_effect_level=step.side_effect_level,
				source_evidence=step.source_evidence,
			)
			for step in draft.steps
		],
	)


def _case(step_count: int = 2) -> WebUITestCase:
	return WebUITestCase(
		root_url='https://example.com/app',
		registrable_domain='example.com',
		steps=[
			WebUITestStep(
				step_id=f'step_{index}',
				instruction=f'Perform action {index}',
				expected_result=f'Expected result {index}',
				expectation_source=ExpectationSource.EXPLICIT,
				source_evidence=[f'Expected result {index}'],
			)
			for index in range(1, step_count + 1)
		],
	)


def _baseline_result(test_case: WebUITestCase) -> QARunResult:
	step_results = [
		QAStepResult(
			step=step,
			status=QAStepStatus.PASSED,
			judgement=StepJudgement(
				action_status=ActionCompletionStatus.COMPLETED,
				expectation_status=ExpectationStatus.MET,
				status=QAStepStatus.PASSED,
				failure_origin=FailureOrigin.NONE,
				reasoning='The expected state was visible.',
				actual_result='The expected state was visible.',
				evidence_ids=['ev_1'],
				replay_assertions=[ReplayAssertion(kind=ReplayAssertionKind.DOM_CONTAINS, value='Expected marker')],
			),
			evidence=StepEvidence(
				before=BrowserEvidenceSnapshot(url='https://example.com/app', dom_summary='Before'),
				after=BrowserEvidenceSnapshot(
					url='https://example.com/app',
					dom_summary='After with Expected marker',
				),
				action_receipt=ActionReceipt(
					status=ActionCompletionStatus.COMPLETED,
					operation_kind=step.operation_kind,
					tool_succeeded=True,
					reasoning='The saved first run completed the business action.',
				),
				artifacts=[EvidenceArtifact(evidence_id='ev_1', kind=EvidenceKind.DOM, summary='Expected marker')],
			),
		)
		for step in test_case.steps
	]
	return QARunResult(
		status=QARunStatus.PASSED,
		failure_origin=FailureOrigin.NONE,
		test_case=test_case,
		step_results=step_results,
		summary='PASSED',
	)


def _qa_llm_for_draft(draft: WebUITestCaseDraft) -> BaseChatModel:
	llm = AsyncMock(spec=BaseChatModel)
	llm.model = 'mock-qa-llm'
	llm.model_name = 'mock-qa-llm'
	llm.name = 'mock-qa-llm'
	llm.provider = 'mock'
	llm._verified_api_keys = True

	async def invoke(_messages, output_format: type[BaseModel] | None = None, **_kwargs):
		assert output_format is not None
		if output_format is WebUITestCaseDraft:
			completion = draft
		elif output_format.__name__ == '_CompiledSteps':
			completion = output_format.model_validate(
				{'steps': [step.model_dump(mode='json') for step in _case_from_draft(draft).steps]}
			)
		else:
			raise AssertionError(f'Unexpected output format: {output_format}')
		return ChatInvokeCompletion(completion=completion, usage=None)

	llm.ainvoke.side_effect = invoke
	return llm


def _setup_fast_run(monkeypatch, agent: Agent, events: list[str]) -> None:
	async def start(_session):
		events.append('browser_start')

	async def finish_execution():
		agent._finalize_qa_result(
			status=QARunStatus.AGENT_FAILED,
			failure_origin=FailureOrigin.AGENT,
			summary='AGENT_FAILED: execution intentionally skipped by callback unit test.',
		)

	monkeypatch.setattr(type(agent.browser_session), 'start', start)
	monkeypatch.setattr(type(agent.browser_session), 'navigate_to', AsyncMock())
	monkeypatch.setattr(agent, '_restore_cached_qa_login_state', AsyncMock())
	monkeypatch.setattr(agent, '_wait_for_qa_page_stability', AsyncMock())
	monkeypatch.setattr(agent, '_execute_initial_actions', AsyncMock())
	monkeypatch.setattr(agent, '_register_skills_as_actions', AsyncMock())
	monkeypatch.setattr(agent, '_start_qa_execution', finish_execution)
	monkeypatch.setattr(agent, '_cache_qa_login_state', AsyncMock())
	monkeypatch.setattr(agent, 'close', AsyncMock())

	class _FakeEvidenceMonitor:
		def __init__(self, *_args, **_kwargs):
			pass

		async def start(self):
			return None

		def cursor(self):
			return None

	monkeypatch.setattr(agent_service, 'QAEvidenceMonitor', _FakeEvidenceMonitor)


@pytest.mark.asyncio
async def test_qa_plan_callback_order_and_ready_before_browser_start(monkeypatch):
	events: list[str] = []
	snapshots: list[QAPlanSnapshot] = []

	def callback(snapshot: QAPlanSnapshot) -> None:
		events.append(snapshot.status)
		snapshots.append(snapshot)

	agent = Agent(
		task=_SIMPLE_TASK,
		llm=_qa_llm_for_draft(_draft_with_steps()),
		register_qa_plan_callback=callback,
	)
	_setup_fast_run(monkeypatch, agent, events)

	await agent.run(max_steps=1)

	assert events == ['generating', 'ready', 'browser_start', 'final']
	assert [snapshot.status for snapshot in snapshots] == ['generating', 'ready', 'final']
	ready = snapshots[1]
	assert [step.step_id for step in ready.steps] == ['step_1', 'step_2']
	assert ready.steps[0].step_num == 1
	assert ready.steps[0].instruction == 'Perform action 1'
	assert ready.steps[0].expected_result == 'Expected result 1'
	assert ready.preconditions[-1] == 'Password: P@ssw0rd123..'
	assert ready.needs_exploration is False


@pytest.mark.asyncio
async def test_qa_plan_async_callback_works(monkeypatch):
	events: list[str] = []
	snapshots: list[QAPlanSnapshot] = []

	async def callback(snapshot: QAPlanSnapshot) -> None:
		events.append(snapshot.status)
		snapshots.append(snapshot)

	agent = Agent(
		task=_SIMPLE_TASK,
		llm=_qa_llm_for_draft(_draft_with_steps()),
		register_qa_plan_callback=callback,
	)
	_setup_fast_run(monkeypatch, agent, events)

	await agent.run(max_steps=1)

	assert [snapshot.status for snapshot in snapshots] == ['generating', 'ready', 'final']


@pytest.mark.asyncio
async def test_qa_plan_sync_callback_returning_awaitable_works(monkeypatch):
	snapshots: list[QAPlanSnapshot] = []
	events: list[str] = []

	async def collect(snapshot: QAPlanSnapshot) -> None:
		snapshots.append(snapshot)

	def callback(snapshot: QAPlanSnapshot):
		events.append(snapshot.status)
		return collect(snapshot)

	agent = Agent(
		task=_SIMPLE_TASK,
		llm=_qa_llm_for_draft(_draft_with_steps()),
		register_qa_plan_callback=callback,
	)
	_setup_fast_run(monkeypatch, agent, events)

	await agent.run(max_steps=1)

	assert [snapshot.status for snapshot in snapshots] == ['generating', 'ready', 'final']


@pytest.mark.asyncio
async def test_qa_plan_ready_marks_missing_expected_result_as_needing_exploration(monkeypatch):
	snapshots: list[QAPlanSnapshot] = []
	events: list[str] = []
	agent = Agent(
		task=_SIMPLE_TASK,
		llm=_qa_llm_for_draft(_draft_with_steps(missing_expected=True)),
		register_qa_plan_callback=snapshots.append,
	)
	_setup_fast_run(monkeypatch, agent, events)

	await agent.run(max_steps=1)

	ready = next(snapshot for snapshot in snapshots if snapshot.status == 'ready')
	assert ready.needs_exploration is True
	assert ready.steps[1].expected_result is None


@pytest.mark.asyncio
async def test_qa_plan_failed_callback_for_invalid_spec(monkeypatch):
	snapshots: list[QAPlanSnapshot] = []
	agent = Agent(
		task='Verify the login page without a URL',
		llm=create_mock_llm(),
		register_qa_plan_callback=snapshots.append,
	)
	start = AsyncMock()
	monkeypatch.setattr(type(agent.browser_session), 'start', start)
	monkeypatch.setattr(agent, 'close', AsyncMock())

	history = await agent.run(max_steps=1)

	assert history.qa_result is not None
	assert history.qa_result.status == QARunStatus.INVALID_SPEC
	start.assert_not_awaited()
	assert [snapshot.status for snapshot in snapshots] == ['failed']
	assert snapshots[0].error_message is not None
	assert 'explicit HTTP(S) start URL' in snapshots[0].error_message


@pytest.mark.asyncio
async def test_qa_plan_failed_callback_for_compiler_validation_error(monkeypatch):
	snapshots: list[QAPlanSnapshot] = []
	agent = Agent(
		task=_SIMPLE_TASK,
		llm=create_mock_llm(),
		register_qa_plan_callback=snapshots.append,
	)
	start = AsyncMock()
	monkeypatch.setattr(type(agent.browser_session), 'start', start)
	monkeypatch.setattr(
		agent_service.QATaskCompiler,
		'extract_requirements',
		AsyncMock(side_effect=ValueError('missing expected result evidence')),
	)
	monkeypatch.setattr(agent, 'close', AsyncMock())

	history = await agent.run(max_steps=1)

	assert history.qa_result is not None
	assert history.qa_result.status == QARunStatus.INVALID_SPEC
	start.assert_not_awaited()
	assert [snapshot.status for snapshot in snapshots] == ['generating', 'failed']
	assert snapshots[-1].error_message is not None
	assert 'missing expected result evidence' in snapshots[-1].error_message


@pytest.mark.asyncio
async def test_qa_plan_reused_compiled_case_emits_final_only(monkeypatch):
	events: list[str] = []
	snapshots: list[QAPlanSnapshot] = []
	agent = Agent(
		task=_SIMPLE_TASK,
		llm=create_mock_llm(),
		qa_test_case=_case(),
		register_qa_plan_callback=lambda snapshot: (events.append(snapshot.status), snapshots.append(snapshot)),
	)
	_setup_fast_run(monkeypatch, agent, events)

	await agent.run(max_steps=1)

	assert events == ['final', 'browser_start']
	assert [snapshot.status for snapshot in snapshots] == ['final']
	assert [step.step_id for step in snapshots[0].steps] == ['step_1', 'step_2']


@pytest.mark.asyncio
async def test_qa_plan_replay_bundle_emits_final_only(monkeypatch, tmp_path):
	events: list[str] = []
	snapshots: list[QAPlanSnapshot] = []
	test_case = _case()
	first_agent = Agent(task='Open https://example.com/app and verify the result.', llm=create_mock_llm())
	first_agent._qa_test_case = test_case
	first_agent._qa_compiled_task = first_agent._qa_original_task
	first_agent._qa_replay_history = {step.step_id: [] for step in test_case.steps}
	first_agent.history.qa_result = _baseline_result(test_case)
	bundle = first_agent.save_qa_bundle(tmp_path / 'bundle')

	second_agent = Agent(
		task='Open https://example.com/app and verify the result.',
		llm=create_mock_llm(),
		register_qa_plan_callback=lambda snapshot: (events.append(snapshot.status), snapshots.append(snapshot)),
	)

	async def start(_session):
		events.append('browser_start')

	monkeypatch.setattr(type(second_agent.browser_session), 'start', start)
	monkeypatch.setattr(second_agent, '_restore_cached_qa_login_state', AsyncMock())
	monkeypatch.setattr(second_agent, '_wait_for_qa_page_stability', AsyncMock())
	monkeypatch.setattr(type(second_agent.browser_session), 'navigate_to', AsyncMock())
	monkeypatch.setattr(second_agent, '_cache_qa_login_state', AsyncMock())
	monkeypatch.setattr(second_agent, 'close', AsyncMock())

	history = await second_agent.rerun(mode='replay', bundle=bundle.path, allow_llm_fallback=False)

	assert history.qa_result is not None
	assert events[:2] == ['final', 'browser_start']
	assert [snapshot.status for snapshot in snapshots] == ['final']


@pytest.mark.asyncio
async def test_qa_plan_callback_exception_does_not_interrupt_agent(monkeypatch):
	events: list[str] = []

	def callback(snapshot: QAPlanSnapshot) -> None:
		events.append(snapshot.status)
		raise RuntimeError('display layer failed')

	agent = Agent(
		task=_SIMPLE_TASK,
		llm=_qa_llm_for_draft(_draft_with_steps()),
		register_qa_plan_callback=callback,
	)
	_setup_fast_run(monkeypatch, agent, events)

	with patch.object(agent.logger, 'warning') as log_warning:
		await agent.run(max_steps=1)

	assert events == ['generating', 'ready', 'browser_start', 'final']
	assert any('QA plan callback failed' in str(call.args[0]) for call in log_warning.call_args_list)


@pytest.mark.asyncio
async def test_no_qa_plan_callback_preserves_existing_run_path(monkeypatch):
	events: list[str] = []
	agent = Agent(
		task=_SIMPLE_TASK,
		llm=_qa_llm_for_draft(_draft_with_steps()),
	)
	_setup_fast_run(monkeypatch, agent, events)

	history = await agent.run(max_steps=1)

	assert history.qa_result is not None
	assert history.qa_result.status == QARunStatus.AGENT_FAILED
	assert events == ['browser_start']


@pytest.mark.asyncio
async def test_article_publish_ready_plan_regression(monkeypatch):
	snapshots: list[QAPlanSnapshot] = []
	events: list[str] = []
	agent = Agent(
		task=_article_task(),
		llm=_qa_llm_for_draft(_article_draft()),
		register_qa_plan_callback=snapshots.append,
	)
	_setup_fast_run(monkeypatch, agent, events)

	await agent.run(max_steps=1)

	ready = next(snapshot for snapshot in snapshots if snapshot.status == 'ready')
	assert len(ready.steps) == 4
	assert [step.step_id for step in ready.steps] == ['step_1', 'step_2', 'step_3', 'step_4']
	assert all(step.instruction for step in ready.steps)
	assert all(step.expected_result for step in ready.steps)
	assert ready.needs_exploration is False
	assert any('P@ssw0rd123..' in precondition for precondition in ready.preconditions)
	assert ready.steps[2].operation_kind == StepOperationKind.SUBMIT
	assert ready.steps[3].side_effect_level == 'irreversible'
