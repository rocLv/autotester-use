"""Metrics for continuously evaluating QA attribution quality and runtime cost."""

from __future__ import annotations

from pydantic import BaseModel, Field

from browser_use.qa.views import QARunResult, QARunStatus


class QAEvaluationSample(BaseModel):
	"""One labelled run used by the QA attribution benchmark."""

	expected_status: QARunStatus
	result: QARunResult
	agent_failure_recovered: bool = False
	navigation_count: int = Field(default=0, ge=0)
	token_count: int = Field(default=0, ge=0)


class QAEvaluationMetrics(BaseModel):
	"""Aggregate reliability, recovery, and efficiency metrics."""

	sample_count: int = Field(ge=0)
	sut_defect_detection_rate: float = Field(ge=0, le=1)
	sut_false_positive_rate: float = Field(ge=0, le=1)
	agent_failure_recovery_rate: float = Field(ge=0, le=1)
	blocked_inconclusive_accuracy: float = Field(ge=0, le=1)
	average_llm_calls: float = Field(ge=0)
	average_tokens: float = Field(ge=0)
	average_navigation_count: float = Field(ge=0)
	average_duration_seconds: float = Field(ge=0)


def _ratio(numerator: float | int, denominator: int) -> float:
	return numerator / denominator if denominator else 0.0


def evaluate_qa_runs(samples: list[QAEvaluationSample]) -> QAEvaluationMetrics:
	"""Calculate release-gate metrics from an explicitly labelled fault corpus."""

	sut_samples = [sample for sample in samples if sample.expected_status == QARunStatus.SUT_FAILED]
	non_sut_samples = [sample for sample in samples if sample.expected_status != QARunStatus.SUT_FAILED]
	agent_samples = [sample for sample in samples if sample.expected_status == QARunStatus.AGENT_FAILED]
	uncertain_samples = [
		sample for sample in samples if sample.expected_status in {QARunStatus.BLOCKED, QARunStatus.INCONCLUSIVE}
	]
	total_duration = sum(sum(timing.elapsed_seconds for timing in sample.result.phase_timings) for sample in samples)
	return QAEvaluationMetrics(
		sample_count=len(samples),
		sut_defect_detection_rate=_ratio(
			sum(sample.result.status == QARunStatus.SUT_FAILED for sample in sut_samples), len(sut_samples)
		),
		sut_false_positive_rate=_ratio(
			sum(sample.result.status == QARunStatus.SUT_FAILED for sample in non_sut_samples), len(non_sut_samples)
		),
		agent_failure_recovery_rate=_ratio(sum(sample.agent_failure_recovered for sample in agent_samples), len(agent_samples)),
		blocked_inconclusive_accuracy=_ratio(
			sum(sample.result.status == sample.expected_status for sample in uncertain_samples), len(uncertain_samples)
		),
		average_llm_calls=_ratio(sum(sample.result.llm_call_count for sample in samples), len(samples)),
		average_tokens=_ratio(sum(sample.token_count for sample in samples), len(samples)),
		average_navigation_count=_ratio(sum(sample.navigation_count for sample in samples), len(samples)),
		average_duration_seconds=_ratio(total_duration, len(samples)),
	)
