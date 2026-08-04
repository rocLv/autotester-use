from browser_use.qa.evaluation import QAEvaluationSample, evaluate_qa_runs
from browser_use.qa.views import FailureOrigin, QARunResult, QARunStatus


def _result(status: QARunStatus, llm_calls: int = 0) -> QARunResult:
	origin = {
		QARunStatus.PASSED: FailureOrigin.NONE,
		QARunStatus.SUT_FAILED: FailureOrigin.SUT,
		QARunStatus.AGENT_FAILED: FailureOrigin.AGENT,
		QARunStatus.BLOCKED: FailureOrigin.ENVIRONMENT,
		QARunStatus.INCONCLUSIVE: FailureOrigin.UNKNOWN,
		QARunStatus.INVALID_SPEC: None,
	}[status]
	return QARunResult(status=status, failure_origin=origin, summary=status.value, llm_call_count=llm_calls)


def test_evaluation_measures_false_positive_and_attribution_accuracy() -> None:
	samples = [
		QAEvaluationSample(expected_status=QARunStatus.SUT_FAILED, result=_result(QARunStatus.SUT_FAILED, 3)),
		QAEvaluationSample(expected_status=QARunStatus.PASSED, result=_result(QARunStatus.SUT_FAILED, 2)),
		QAEvaluationSample(expected_status=QARunStatus.BLOCKED, result=_result(QARunStatus.BLOCKED, 1)),
		QAEvaluationSample(
			expected_status=QARunStatus.AGENT_FAILED,
			result=_result(QARunStatus.PASSED, 2),
			agent_failure_recovered=True,
		),
	]

	metrics = evaluate_qa_runs(samples)

	assert metrics.sut_defect_detection_rate == 1
	assert metrics.sut_false_positive_rate == 1 / 3
	assert metrics.agent_failure_recovery_rate == 1
	assert metrics.blocked_inconclusive_accuracy == 1
	assert metrics.average_llm_calls == 2
