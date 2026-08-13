"""Pydantic models for Web UI QA task compilation and execution results."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExpectationSource(StrEnum):
	"""Authority backing a test-step expectation."""

	EXPLICIT = 'explicit'
	UI_CONTRACT = 'ui_contract'
	HEURISTIC = 'heuristic'


class RequirementSource(StrEnum):
	"""Source document that authoritatively defines a requirement."""

	TASK = 'task'
	GROUND_TRUTH = 'ground_truth'
	UI_CONTRACT = 'ui_contract'
	LEGACY = 'legacy'


class StepOperationKind(StrEnum):
	"""Business operation performed by one QA step."""

	OBSERVE = 'observe'
	CLICK = 'click'
	INPUT = 'input'
	NAVIGATE = 'navigate'
	SUBMIT = 'submit'
	OTHER = 'other'


class SideEffectLevel(StrEnum):
	"""Risk that repeating a business step changes external state."""

	NONE = 'none'
	REVERSIBLE = 'reversible'
	IRREVERSIBLE = 'irreversible'


class QAPlanStep(BaseModel):
	"""Public, browser-independent view of one compiled QA plan step."""

	model_config = ConfigDict(extra='forbid')

	step_num: int = Field(ge=1)
	step_id: str = Field(min_length=1)
	instruction: str = Field(min_length=1)
	expected_result: str | None = None
	operation_kind: StepOperationKind
	side_effect_level: SideEffectLevel
	preconditions: list[str] = Field(default_factory=list)
	source_evidence: list[str] = Field(default_factory=list)


class QAPlanSnapshot(BaseModel):
	"""Public snapshot emitted while QA requirements are compiled into executable steps."""

	model_config = ConfigDict(extra='forbid')

	status: Literal['generating', 'ready', 'final', 'failed']
	preconditions: list[str] = Field(default_factory=list)
	steps: list[QAPlanStep] = Field(default_factory=list)
	needs_exploration: bool = False
	error_message: str | None = None


class PreconditionMode(StrEnum):
	"""Whether a precondition is only checked or may be established."""

	VERIFY = 'verify'
	ENSURE = 'ensure'


class PreconditionStatus(StrEnum):
	"""Result of evaluating one typed precondition."""

	SATISFIED = 'satisfied'
	ENSURED = 'ensured'
	BLOCKED = 'blocked'
	NOT_CHECKED = 'not_checked'


class FailureOrigin(StrEnum):
	"""Root cause for a terminal QA outcome."""

	NONE = 'none'
	SUT = 'sut'
	AGENT = 'agent'
	ENVIRONMENT = 'environment'
	UNKNOWN = 'unknown'


class FailureCode(StrEnum):
	"""Stable detailed failure attribution beneath the public top-level status."""

	NONE = 'none'
	SUT_EXPECTATION_MISMATCH = 'sut_expectation_mismatch'
	SUT_RELATED_HTTP_ERROR = 'sut_related_http_error'
	SUT_INVALID_STATE = 'sut_invalid_state'
	AGENT_WRONG_TARGET = 'agent_wrong_target'
	AGENT_ACTION_ERROR = 'agent_action_error'
	AGENT_STEP_BUDGET = 'agent_step_budget'
	AGENT_COMPILER_ERROR = 'agent_compiler_error'
	AGENT_JUDGE_ERROR = 'agent_judge_error'
	AGENT_MODEL_FAILURE = 'agent_model_failure'
	ENV_AUTH_REQUIRED = 'env_auth_required'
	ENV_CAPTCHA = 'env_captcha'
	ENV_BROWSER = 'env_browser'
	ENV_NETWORK = 'env_network'
	ENV_NAVIGATION_POLICY = 'env_navigation_policy'
	ENV_TEST_DATA = 'env_test_data'
	UNKNOWN_CONFLICT = 'unknown_conflict'
	UNKNOWN_INSUFFICIENT_EVIDENCE = 'unknown_insufficient_evidence'
	UNKNOWN_SIDE_EFFECT = 'unknown_side_effect'
	UNKNOWN_OTHER = 'unknown_other'


class QARunStatus(StrEnum):
	"""Terminal status for one Web UI test case."""

	PASSED = 'PASSED'
	SUT_FAILED = 'SUT_FAILED'
	AGENT_FAILED = 'AGENT_FAILED'
	BLOCKED = 'BLOCKED'
	INCONCLUSIVE = 'INCONCLUSIVE'
	INVALID_SPEC = 'INVALID_SPEC'


class QAStepStatus(StrEnum):
	"""Execution status for an individual business test step."""

	PASSED = 'PASSED'
	SUT_FAILED = 'SUT_FAILED'
	AGENT_FAILED = 'AGENT_FAILED'
	BLOCKED = 'BLOCKED'
	INCONCLUSIVE = 'INCONCLUSIVE'
	NOT_RUN = 'NOT_RUN'


class ActionCompletionStatus(StrEnum):
	"""Whether the intended business action actually took effect."""

	COMPLETED = 'completed'
	NOT_COMPLETED = 'not_completed'
	UNCERTAIN = 'uncertain'


class ExpectationStatus(StrEnum):
	"""Whether observed SUT state matches the expected state."""

	MET = 'met'
	NOT_MET = 'not_met'
	NOT_OBSERVABLE = 'not_observable'


class EvidenceKind(StrEnum):
	"""Kinds of objective artifacts that a QA verdict may cite."""

	URL = 'url'
	DOM = 'dom'
	SCREENSHOT = 'screenshot'
	ACTION = 'action'
	NETWORK = 'network'
	CONSOLE = 'console'
	BROWSER = 'browser'
	STORAGE = 'storage'
	DOWNLOAD = 'download'


class EvidenceQuality(StrEnum):
	"""Runner-computed strength of the evidence supporting a verdict."""

	STRONG = 'strong'
	MEDIUM = 'medium'
	WEAK = 'weak'


class ReplayAssertionKind(StrEnum):
	"""Deterministic browser evidence checks that require no model call."""

	URL_EQUALS = 'url_equals'
	URL_CONTAINS = 'url_contains'
	TITLE_CONTAINS = 'title_contains'
	DOM_CONTAINS = 'dom_contains'
	DOM_NOT_CONTAINS = 'dom_not_contains'
	ELEMENT_VISIBLE = 'element_visible'
	ELEMENT_VALUE_EQUALS = 'element_value_equals'
	ATTRIBUTE_EQUALS = 'attribute_equals'
	COUNT_EQUALS = 'count_equals'
	RESPONSE_STATUS = 'response_status'
	JSON_PATH_EQUALS = 'json_path_equals'
	DOWNLOAD_EXISTS = 'download_exists'
	STORAGE_VALUE_EQUALS = 'storage_value_equals'


class ReplayAssertion(BaseModel):
	"""One objective assertion learned from a reliable first-run judgement."""

	model_config = ConfigDict(extra='forbid')

	kind: ReplayAssertionKind
	value: str = Field(min_length=1)
	case_sensitive: bool = True
	description: str = ''
	locator: str | None = None
	attribute: str | None = None
	expected_count: int | None = Field(default=None, ge=0)
	status_code: int | None = Field(default=None, ge=100, le=599)
	json_path: str | None = None
	storage_key: str | None = None

	@field_validator('value')
	@classmethod
	def strip_assertion_value(cls, value: str) -> str:
		value = value.strip()
		if not value:
			raise ValueError('replay assertion value must not be blank')
		return value


class RequirementReference(BaseModel):
	"""Verified provenance for an explicit requirement or discovered UI contract."""

	model_config = ConfigDict(extra='forbid')

	source: RequirementSource
	quote: str = Field(min_length=1)
	start: int | None = Field(default=None, ge=0)
	end: int | None = Field(default=None, ge=0)
	location: str | None = None

	@model_validator(mode='after')
	def validate_span(self) -> RequirementReference:
		self.quote = self.quote.strip()
		if self.source in {RequirementSource.TASK, RequirementSource.GROUND_TRUTH}:
			if self.start is None or self.end is None or self.end <= self.start:
				raise ValueError(f'{self.source.value} requirement references require a valid character span')
		if (self.start is None) != (self.end is None):
			raise ValueError('requirement reference start and end must be supplied together')
		return self


class QAPrecondition(BaseModel):
	"""One typed condition that must hold before formal business-step execution."""

	model_config = ConfigDict(extra='forbid')

	precondition_id: str = Field(min_length=1)
	description: str = Field(min_length=1)
	mode: PreconditionMode = PreconditionMode.VERIFY
	required: bool = True
	sensitive_refs: list[str] = Field(default_factory=list)

	@field_validator('precondition_id', 'description')
	@classmethod
	def strip_precondition_text(cls, value: str) -> str:
		value = value.strip()
		if not value:
			raise ValueError('precondition text must not be blank')
		return value


class QAPreconditionResult(BaseModel):
	"""Recorded outcome for a precondition check or setup action."""

	model_config = ConfigDict(extra='forbid')

	precondition: QAPrecondition
	status: PreconditionStatus
	reason: str = ''
	evidence_ids: list[str] = Field(default_factory=list)


def _migrate_precondition(value: str | QAPrecondition | dict[str, Any], index: int) -> QAPrecondition:
	"""Accept v1 string preconditions while always emitting typed v2 objects."""

	if isinstance(value, QAPrecondition):
		return value
	if isinstance(value, str):
		return QAPrecondition(precondition_id=f'precondition_{index + 1}', description=value)
	return QAPrecondition.model_validate(value)


class WebUITestStep(BaseModel):
	"""One user-visible business step and its observable expected result."""

	model_config = ConfigDict(extra='forbid')

	step_id: str = Field(min_length=1, description='Stable identifier unique inside the test case')
	instruction: str = Field(min_length=1, description='User-visible business action to perform')
	expected_result: str = Field(min_length=1, description='Observable state that must be present after the action')
	expectation_source: ExpectationSource
	operation_kind: StepOperationKind = StepOperationKind.OTHER
	side_effect_level: SideEffectLevel = SideEffectLevel.NONE
	idempotency_key: str | None = Field(
		default=None,
		description='Run-scoped identifier exposed to side-effecting steps; ${run_id} is resolved by the runner',
	)
	requirement_references: list[RequirementReference] = Field(default_factory=list)
	source_evidence: list[str] = Field(
		default_factory=list,
		description='Deprecated v1 compatibility view of requirement reference quotes',
	)
	preconditions: list[QAPrecondition] = Field(default_factory=list)

	@field_validator('preconditions', mode='before')
	@classmethod
	def migrate_step_preconditions(cls, value: Any) -> list[QAPrecondition]:
		return [_migrate_precondition(item, index) for index, item in enumerate(value or [])]

	@field_validator('step_id', 'instruction', 'expected_result')
	@classmethod
	def strip_required_text(cls, value: str) -> str:
		value = value.strip()
		if not value:
			raise ValueError('value must not be blank')
		return value

	@model_validator(mode='after')
	def validate_expectation_authority(self) -> WebUITestStep:
		if self.side_effect_level != SideEffectLevel.NONE and not self.idempotency_key:
			self.idempotency_key = f'autotester:{self.step_id}:${{run_id}}'
		if not self.requirement_references and self.source_evidence:
			reference_source = (
				RequirementSource.UI_CONTRACT
				if self.expectation_source == ExpectationSource.UI_CONTRACT
				else RequirementSource.LEGACY
			)
			self.requirement_references = [
				RequirementReference(source=reference_source, quote=quote, location='v1 source_evidence')
				for quote in self.source_evidence
			]
		if self.requirement_references and not self.source_evidence:
			self.source_evidence = [reference.quote for reference in self.requirement_references]
		if (
			self.expectation_source in {ExpectationSource.EXPLICIT, ExpectationSource.UI_CONTRACT}
			and not self.requirement_references
		):
			raise ValueError(f'{self.expectation_source} expectations require requirement_references')
		return self


class WebUITestCase(BaseModel):
	"""Strict structured representation compiled from a natural-language QA task."""

	model_config = ConfigDict(extra='forbid')

	root_url: str
	registrable_domain: str
	preconditions: list[QAPrecondition] = Field(default_factory=list)
	steps: list[WebUITestStep] = Field(min_length=1)
	cleanup_steps: list[WebUITestStep] = Field(default_factory=list)
	test_data: dict[str, str] = Field(default_factory=dict)

	@field_validator('preconditions', mode='before')
	@classmethod
	def migrate_case_preconditions(cls, value: Any) -> list[QAPrecondition]:
		return [_migrate_precondition(item, index) for index, item in enumerate(value or [])]

	@field_validator('root_url')
	@classmethod
	def validate_root_url(cls, value: str) -> str:
		value = value.strip()
		parsed = urlparse(value)
		if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
			raise ValueError('root_url must be an absolute HTTP(S) URL')
		if parsed.username or parsed.password:
			raise ValueError('root_url must not contain credentials')
		return value

	@model_validator(mode='after')
	def validate_unique_step_ids(self) -> WebUITestCase:
		step_ids = [step.step_id for step in self.steps]
		if len(step_ids) != len(set(step_ids)):
			raise ValueError('step_id values must be unique')
		return self

	@staticmethod
	def _markdown_cell(value: str | list[str]) -> str:
		"""Escape a scalar or list for safe use inside a Markdown table cell."""

		text = '<br>'.join(value) if isinstance(value, list) else value
		return text.replace('\\', '\\\\').replace('|', '\\|').replace('\r\n', '<br>').replace('\n', '<br>') or '—'

	@staticmethod
	def _precondition_descriptions(items: list[Any]) -> list[str]:
		return [
			f'{getattr(getattr(item, "mode", None), "value", "verify").upper()} '
			f'{getattr(item, "precondition_id", f"precondition_{index + 1}")}: {getattr(item, "description", item)}'
			for index, item in enumerate(items)
		]

	def to_markdown_table(self) -> str:
		"""Render the strongly typed QA document as readable Markdown tables."""

		cell = self._markdown_cell
		lines = [
			'| 用例字段 | 值 |',
			'| --- | --- |',
			f'| 起始 URL | {cell(self.root_url)} |',
			f'| 注册主域 | {cell(self.registrable_domain)} |',
			f'| 全局前置条件 | {cell(self._precondition_descriptions(self.preconditions))} |',
			'',
			'| 步骤 ID | 步骤前置条件 | 操作类型 | 副作用级别 | 幂等标识 | 操作 | 预期结果 | 预期来源 | 需求引用 |',
			'| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
		]
		for step in self.steps:
			lines.append(
				f'| {cell(step.step_id)} | {cell(self._precondition_descriptions(step.preconditions))} | '
				f'{cell(step.operation_kind.value)} | {cell(step.side_effect_level.value)} | '
				f'{cell(step.idempotency_key or "—")} | {cell(step.instruction)} | '
				f'{cell(step.expected_result)} | {cell(step.expectation_source.value)} | '
				f'{cell([reference.quote for reference in step.requirement_references])} |'
			)
		if self.cleanup_steps:
			lines.extend(['', '| Cleanup ID | 操作 | 预期结果 | 副作用级别 |', '| --- | --- | --- | --- |'])
			for step in self.cleanup_steps:
				lines.append(
					f'| {cell(step.step_id)} | {cell(step.instruction)} | {cell(step.expected_result)} | '
					f'{cell(step.side_effect_level.value)} |'
				)
		return '\n'.join(lines)


class FinishTestStepAction(BaseModel):
	"""Executor signal that a business step is ready for independent judgement."""

	actual_result: str = Field(min_length=1, description='Objective UI state observed after attempting the step')
	evidence: list[str] = Field(
		default_factory=list,
		description='Concise DOM, URL, message, or visual observations; do not include a pass/fail opinion',
	)
	action_completed: bool | None = Field(
		default=None,
		description='Executor observation only; the independent judge verifies this against browser evidence',
	)
	side_effect_uncertain: bool = Field(
		default=False,
		description='True when a submit/delete/payment-like action may have happened and must not be blindly retried',
	)


class EvidenceArtifact(BaseModel):
	"""One immutable, machine-addressable QA evidence item."""

	model_config = ConfigDict(extra='forbid')

	evidence_id: str = Field(default_factory=lambda: f'ev_{uuid4().hex}', min_length=1)
	kind: EvidenceKind
	timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
	tab_id: str | None = None
	frame_id: str | None = None
	url: str | None = None
	summary: str = ''
	artifact_path: str | None = None
	sha256: str | None = None
	redacted: bool = True
	metadata: dict[str, Any] = Field(default_factory=dict)

	@field_validator('artifact_path')
	@classmethod
	def normalize_artifact_path(cls, value: str | None) -> str | None:
		return str(Path(value)) if value else None


class ActionTargetProof(BaseModel):
	"""Tool-authored, objectively verified proof that a custom action used its intended target."""

	model_config = ConfigDict(extra='forbid')

	target_name: str = Field(min_length=1, description='Non-sensitive name of the target controlled by the tool')
	target_matched: bool
	verification: dict[str, bool] = Field(
		min_length=1,
		description='Boolean checks performed by the tool against the live browser state',
	)

	@model_validator(mode='after')
	def validate_target_match(self) -> ActionTargetProof:
		if self.target_matched != all(self.verification.values()):
			raise ValueError('target_matched must equal the conjunction of verification checks')
		return self


class ActionExpectationProof(BaseModel):
	"""Tool-authored objective checks tied to an exact verified requirement quote."""

	model_config = ConfigDict(extra='forbid')

	requirement_quote: str = Field(min_length=1)
	expectation_met: bool
	verification: dict[str, bool] = Field(min_length=1)

	@model_validator(mode='after')
	def validate_expectation(self) -> ActionExpectationProof:
		if self.expectation_met != all(self.verification.values()):
			raise ValueError('expectation_met must equal the conjunction of verification checks')
		return self


class ActionReceipt(BaseModel):
	"""Runner-owned proof that the intended low-level business action was attempted correctly."""

	model_config = ConfigDict(extra='forbid')

	status: ActionCompletionStatus
	operation_kind: StepOperationKind
	tool_succeeded: bool
	target_matched: bool | None = None
	selected_element: dict[str, Any] | None = None
	input_values: list[str] = Field(default_factory=list)
	state_changed: bool | None = None
	side_effect_uncertain: bool = False
	action_names: list[str] = Field(default_factory=list)
	related_request_ids: list[str] = Field(default_factory=list)
	evidence_ids: list[str] = Field(default_factory=list)
	reasoning: str = Field(min_length=1)


class BrowserEvidenceSnapshot(BaseModel):
	"""Compact, objective browser state captured at a business-step boundary."""

	url: str
	title: str = ''
	dom_summary: str = ''
	screenshot_path: str | None = None
	browser_errors: list[str] = Field(default_factory=list)
	network_errors: list[str] = Field(default_factory=list)
	console_errors: list[str] = Field(default_factory=list)
	recent_events: str | None = None
	artifacts: list[EvidenceArtifact] = Field(default_factory=list)

	@property
	def evidence_ids(self) -> list[str]:
		return [artifact.evidence_id for artifact in self.artifacts]


class StepEvidence(BaseModel):
	"""Evidence supplied to an independent step judge."""

	before: BrowserEvidenceSnapshot
	after: BrowserEvidenceSnapshot
	action_results: list[dict[str, Any]] = Field(default_factory=list)
	action_receipt: ActionReceipt | None = None
	artifacts: list[EvidenceArtifact] = Field(default_factory=list)
	executor_report: str | None = None
	side_effect_uncertain: bool = False
	evidence_quality: EvidenceQuality = EvidenceQuality.WEAK

	@model_validator(mode='after')
	def collect_snapshot_artifacts(self) -> StepEvidence:
		known = {artifact.evidence_id for artifact in self.artifacts}
		for artifact in [*self.before.artifacts, *self.after.artifacts]:
			if artifact.evidence_id not in known:
				self.artifacts.append(artifact)
				known.add(artifact.evidence_id)
		return self

	@property
	def evidence_ids(self) -> set[str]:
		return {artifact.evidence_id for artifact in self.artifacts}


class StepJudgement(BaseModel):
	"""Independent judgement of one business step."""

	model_config = ConfigDict(extra='forbid')

	action_status: ActionCompletionStatus
	expectation_status: ExpectationStatus
	status: QAStepStatus
	failure_origin: FailureOrigin
	failure_code: FailureCode = FailureCode.NONE
	reasoning: str = Field(min_length=1)
	actual_result: str = Field(min_length=1)
	evidence: list[str] = Field(default_factory=list)
	evidence_ids: list[str] = Field(default_factory=list)
	confidence: float = Field(default=0.5, ge=0, le=1)
	replay_assertions: list[ReplayAssertion] = Field(
		default_factory=list,
		description='Objective URL/title/DOM checks that can revalidate this PASSED step without an LLM',
	)
	retry_safe: bool = False

	@model_validator(mode='after')
	def validate_consistency(self) -> StepJudgement:
		if self.status == QAStepStatus.NOT_RUN:
			raise ValueError('NOT_RUN is assigned by the runner and cannot be returned by the judge')
		if self.status == QAStepStatus.PASSED:
			if self.action_status != ActionCompletionStatus.COMPLETED or self.expectation_status != ExpectationStatus.MET:
				raise ValueError('PASSED requires a completed action and a met expectation')
			if self.failure_origin != FailureOrigin.NONE:
				raise ValueError('PASSED cannot have a failure origin')
			self.failure_code = FailureCode.NONE
		elif self.status == QAStepStatus.SUT_FAILED:
			if self.action_status != ActionCompletionStatus.COMPLETED:
				raise ValueError('SUT_FAILED requires objective evidence that the action completed')
			if self.expectation_status != ExpectationStatus.NOT_MET or self.failure_origin != FailureOrigin.SUT:
				raise ValueError('SUT_FAILED requires a non-met expectation attributed to the SUT')
		elif self.status == QAStepStatus.AGENT_FAILED and self.failure_origin != FailureOrigin.AGENT:
			raise ValueError('AGENT_FAILED must be attributed to the agent')
		elif self.status == QAStepStatus.BLOCKED and self.failure_origin != FailureOrigin.ENVIRONMENT:
			raise ValueError('BLOCKED must be attributed to the environment')
		elif self.status == QAStepStatus.INCONCLUSIVE and self.failure_origin != FailureOrigin.UNKNOWN:
			raise ValueError('INCONCLUSIVE must use unknown failure origin')
		return self


class ReviewRecord(BaseModel):
	"""Independent risk review performed without exposing the primary verdict."""

	model_config = ConfigDict(extra='forbid')

	required: bool = True
	primary_status: QAStepStatus
	secondary_status: QAStepStatus | None = None
	agreed: bool = False
	reviewer_model: str | None = None
	reason: str = ''
	evidence_ids: list[str] = Field(default_factory=list)


class QAStepResult(BaseModel):
	"""Recorded execution result for a test step."""

	step: WebUITestStep
	status: QAStepStatus
	retry_count: int = Field(default=0, ge=0, le=3)
	judgement: StepJudgement | None = None
	review: ReviewRecord | None = None
	evidence: StepEvidence | None = None
	attempt_receipts: list[ActionReceipt] = Field(default_factory=list)

	@property
	def has_reliable_verdict(self) -> bool:
		"""Whether receipt, evidence, and any required SUT review prove this step."""

		if self.status not in {QAStepStatus.PASSED, QAStepStatus.SUT_FAILED}:
			return False
		if self.judgement is None or self.evidence is None or self.evidence.action_receipt is None:
			return False
		if self.evidence.action_receipt.status != ActionCompletionStatus.COMPLETED:
			return False
		if not self.judgement.evidence_ids or not set(self.judgement.evidence_ids) <= self.evidence.evidence_ids:
			return False
		if self.status == QAStepStatus.SUT_FAILED:
			return bool(self.review and self.review.agreed and self.review.secondary_status == QAStepStatus.SUT_FAILED)
		return True


class QACleanupResult(BaseModel):
	"""Cleanup outcome recorded separately from the business verdict."""

	step: WebUITestStep
	status: QAStepStatus
	reason: str
	evidence_ids: list[str] = Field(default_factory=list)


class QAPhaseTiming(BaseModel):
	"""Measured wall-clock time for one named QA runner phase."""

	model_config = ConfigDict(extra='forbid')

	phase: str = Field(min_length=1)
	elapsed_seconds: float = Field(ge=0)


class QARunResult(BaseModel):
	"""Terminal Web UI QA result attached to ``AgentHistoryList``."""

	schema_version: Literal[2] = 2
	run_id: str = Field(default_factory=lambda: str(uuid4()))
	legacy_imported: bool = False
	status: QARunStatus
	failure_origin: FailureOrigin | None
	failure_code: FailureCode = FailureCode.NONE
	test_case: WebUITestCase | None = None
	precondition_results: list[QAPreconditionResult] = Field(default_factory=list)
	step_results: list[QAStepResult] = Field(default_factory=list)
	cleanup_results: list[QACleanupResult] = Field(default_factory=list)
	stopped_at_step: str | None = None
	summary: str
	validation_errors: list[str] = Field(default_factory=list)
	warnings: list[str] = Field(default_factory=list)
	requested_mode: Literal['ai', 'replay'] = 'ai'
	effective_mode: Literal['ai', 'replay', 'ai_fallback'] = 'ai'
	llm_call_count: int = Field(default=0, ge=0)
	environment: dict[str, Any] = Field(default_factory=dict)
	artifacts: list[EvidenceArtifact] = Field(default_factory=list)
	phase_timings: list[QAPhaseTiming] = Field(
		default_factory=list,
		description='Wall-clock timings emitted by the QA runner for performance diagnosis',
	)

	@model_validator(mode='after')
	def validate_status_origin(self) -> QARunResult:
		expected_origin = {
			QARunStatus.PASSED: FailureOrigin.NONE,
			QARunStatus.SUT_FAILED: FailureOrigin.SUT,
			QARunStatus.AGENT_FAILED: FailureOrigin.AGENT,
			QARunStatus.BLOCKED: FailureOrigin.ENVIRONMENT,
			QARunStatus.INCONCLUSIVE: FailureOrigin.UNKNOWN,
		}
		if self.status in expected_origin and self.failure_origin != expected_origin[self.status]:
			raise ValueError(f'{self.status} requires failure_origin={expected_origin[self.status]}')
		if self.status == QARunStatus.INVALID_SPEC and self.failure_origin is not None:
			raise ValueError('INVALID_SPEC cannot have a failure origin')
		if self.status == QARunStatus.PASSED:
			self.failure_code = FailureCode.NONE
		return self

	@property
	def has_reliable_verdict(self) -> bool:
		"""Whether this run produced a reliable product pass/fail verdict."""

		if self.legacy_imported or self.status not in {QARunStatus.PASSED, QARunStatus.SUT_FAILED}:
			return False
		reliable_steps = [item for item in self.step_results if item.status in {QAStepStatus.PASSED, QAStepStatus.SUT_FAILED}]
		if not reliable_steps:
			return False
		return all(item.has_reliable_verdict for item in reliable_steps)
