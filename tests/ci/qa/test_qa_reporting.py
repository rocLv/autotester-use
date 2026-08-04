from pathlib import Path
from xml.etree import ElementTree

from browser_use.qa.reporting import qa_html_report, qa_junit_xml, write_qa_json_report
from browser_use.qa.views import (
	ExpectationSource,
	FailureOrigin,
	QARunResult,
	QARunStatus,
	QAStepResult,
	QAStepStatus,
	WebUITestCase,
	WebUITestStep,
)


def _result_with_all_step_outcomes() -> QARunResult:
	statuses = [
		QAStepStatus.SUT_FAILED,
		QAStepStatus.AGENT_FAILED,
		QAStepStatus.INCONCLUSIVE,
		QAStepStatus.NOT_RUN,
	]
	steps = [
		WebUITestStep(
			step_id=f'step-{index}',
			instruction=f'Action {index}',
			expected_result=f'Expected {index}',
			expectation_source=ExpectationSource.EXPLICIT,
			source_evidence=[f'Expected {index}'],
		)
		for index in range(len(statuses))
	]
	case = WebUITestCase(root_url='https://example.com/app', registrable_domain='example.com', steps=steps)
	return QARunResult(
		status=QARunStatus.SUT_FAILED,
		failure_origin=FailureOrigin.SUT,
		test_case=case,
		step_results=[QAStepResult(step=step, status=status) for step, status in zip(steps, statuses, strict=True)],
		summary='<script>alert("escaped")</script>',
	)


def test_junit_maps_product_failures_errors_and_not_run() -> None:
	root = ElementTree.fromstring(qa_junit_xml(_result_with_all_step_outcomes()))

	assert root.attrib == {
		'name': 'AutoTester Use Web UI QA',
		'tests': '4',
		'failures': '1',
		'errors': '2',
		'skipped': '1',
	}
	assert len(root.findall('.//failure')) == 1
	assert len(root.findall('.//error')) == 2
	assert len(root.findall('.//skipped')) == 1


def test_junit_records_run_level_block_before_business_steps() -> None:
	result = QARunResult(
		status=QARunStatus.BLOCKED,
		failure_origin=FailureOrigin.ENVIRONMENT,
		summary='Required login credentials are unavailable.',
	)
	root = ElementTree.fromstring(qa_junit_xml(result))
	testcase = root.find('.//testcase')
	error = root.find('.//error')

	assert root.attrib['tests'] == '1'
	assert root.attrib['errors'] == '1'
	assert testcase is not None
	assert error is not None
	assert testcase.attrib['name'] == '__run__'
	assert error.attrib['type'] == 'blocked'


def test_html_escapes_untrusted_diagnostics_and_json_is_typed(tmp_path: Path) -> None:
	result = _result_with_all_step_outcomes()
	html = qa_html_report(result)

	assert '<script>alert' not in html
	assert '&lt;script&gt;' in html
	json_path = write_qa_json_report(result, tmp_path / 'report.json')
	assert '"schema_version": 2' in json_path.read_text(encoding='utf-8')
