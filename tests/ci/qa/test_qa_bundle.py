from pathlib import Path

import pytest

from browser_use.qa.bundle import QABundle
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
	QARunResult,
	QARunStatus,
	QAStepResult,
	QAStepStatus,
	RequirementReference,
	RequirementSource,
	StepEvidence,
	StepJudgement,
	WebUITestCase,
	WebUITestStep,
)


def _passed_result(tmp_path: Path) -> tuple[str, QARunResult]:
	task = 'Open https://example.com/app and expect the “Ready” heading.'
	quote = 'expect the “Ready” heading'
	start = task.index(quote)
	step = WebUITestStep(
		step_id='heading',
		instruction='Inspect the heading',
		expected_result='The “Ready” heading is visible',
		expectation_source=ExpectationSource.EXPLICIT,
		requirement_references=[
			RequirementReference(
				source=RequirementSource.TASK,
				quote=quote,
				start=start,
				end=start + len(quote),
			)
		],
	)
	case = WebUITestCase(root_url='https://example.com/app', registrable_domain='example.com', steps=[step])
	artifact_file = tmp_path / 'redacted-dom.txt'
	artifact_file.write_text('<h1>Ready</h1>', encoding='utf-8')
	artifact = EvidenceArtifact(
		kind=EvidenceKind.DOM,
		summary='Ready heading',
		artifact_path=str(artifact_file),
		redacted=True,
	)
	before = BrowserEvidenceSnapshot(url=case.root_url)
	after = BrowserEvidenceSnapshot(url=case.root_url, dom_summary='Ready', artifacts=[artifact])
	receipt = ActionReceipt(
		status=ActionCompletionStatus.COMPLETED,
		operation_kind=step.operation_kind,
		tool_succeeded=True,
		evidence_ids=[artifact.evidence_id],
		reasoning='Observation completed.',
	)
	evidence = StepEvidence(
		before=before,
		after=after,
		action_receipt=receipt,
		evidence_quality=EvidenceQuality.STRONG,
	)
	judgement = StepJudgement(
		action_status=ActionCompletionStatus.COMPLETED,
		expectation_status=ExpectationStatus.MET,
		status=QAStepStatus.PASSED,
		failure_origin=FailureOrigin.NONE,
		reasoning='The heading is present.',
		actual_result='Ready is visible.',
		evidence_ids=[artifact.evidence_id],
		confidence=1,
	)
	return task, QARunResult(
		status=QARunStatus.PASSED,
		failure_origin=FailureOrigin.NONE,
		test_case=case,
		step_results=[QAStepResult(step=step, status=QAStepStatus.PASSED, judgement=judgement, evidence=evidence)],
		summary='PASSED',
		artifacts=[artifact],
		environment={'headless': True, 'browser': 'chromium'},
	)


def test_bundle_round_trip_verifies_files_and_resolves_artifact_paths(tmp_path: Path) -> None:
	task, result = _passed_result(tmp_path)
	bundle = QABundle.save(tmp_path / 'bundle', task=task, ground_truth=None, run_result=result, action_history={})

	loaded = QABundle.load(bundle.path)

	assert loaded.manifest.schema_version == 2
	assert loaded.manifest.environment == result.environment
	assert loaded.test_case == result.test_case
	assert loaded.run_result.status == QARunStatus.PASSED
	assert Path(loaded.run_result.artifacts[0].artifact_path or '').is_file()


def test_bundle_rejects_tampering_and_identity_mismatch(tmp_path: Path) -> None:
	task, result = _passed_result(tmp_path)
	bundle = QABundle.save(tmp_path / 'bundle', task=task, ground_truth=None, run_result=result, action_history={})
	revision_root = bundle.path / 'revisions' / bundle.revision.revision_id
	(revision_root / 'test_case.json').write_text('{}', encoding='utf-8')

	with pytest.raises(ValueError, match='integrity check failed'):
		QABundle.load(bundle.path)

	other_path = tmp_path / 'other-bundle'
	QABundle.save(other_path, task=task, ground_truth=None, run_result=result, action_history={})
	with pytest.raises(ValueError, match='different Task'):
		QABundle.save(
			other_path,
			task='Open https://example.com/app with a different Task',
			ground_truth=None,
			run_result=result,
			action_history={},
		)
