"""Deterministic JSON, JUnit XML, and self-contained HTML QA reports."""

from __future__ import annotations

import html
import json
from pathlib import Path
from xml.etree import ElementTree

from browser_use.qa.views import QARunResult, QARunStatus, QAStepStatus


def _write_text(path: str | Path, content: str) -> Path:
	report_path = Path(path).expanduser().resolve()
	report_path.parent.mkdir(parents=True, exist_ok=True)
	temporary_path = report_path.with_name(f'.{report_path.name}.tmp')
	temporary_path.write_text(content, encoding='utf-8')
	temporary_path.replace(report_path)
	return report_path


def write_qa_json_report(result: QARunResult, path: str | Path) -> Path:
	"""Write the complete strongly typed result as UTF-8 JSON."""

	return _write_text(path, json.dumps(result.model_dump(mode='json'), ensure_ascii=False, indent=2))


def qa_junit_xml(result: QARunResult) -> str:
	"""Render JUnit semantics: product defects fail, infrastructure outcomes error."""

	has_step_results = bool(result.step_results)
	tests = len(result.step_results) if has_step_results else 1
	failures = sum(item.status == QAStepStatus.SUT_FAILED for item in result.step_results)
	errors = sum(
		item.status in {QAStepStatus.AGENT_FAILED, QAStepStatus.BLOCKED, QAStepStatus.INCONCLUSIVE}
		for item in result.step_results
	)
	skipped = sum(item.status == QAStepStatus.NOT_RUN for item in result.step_results)
	if not has_step_results:
		failures = int(result.status == QARunStatus.SUT_FAILED)
		errors = int(
			result.status
			in {
				QARunStatus.AGENT_FAILED,
				QARunStatus.BLOCKED,
				QARunStatus.INCONCLUSIVE,
				QARunStatus.INVALID_SPEC,
			}
		)
	suite = ElementTree.Element(
		'testsuite',
		{
			'name': 'AutoTester Use Web UI QA',
			'tests': str(tests),
			'failures': str(failures),
			'errors': str(errors),
			'skipped': str(skipped),
		},
	)
	properties = ElementTree.SubElement(suite, 'properties')
	for key, value in {
		'schema_version': result.schema_version,
		'run_id': result.run_id,
		'run_status': result.status.value,
		'requested_mode': result.requested_mode,
		'effective_mode': result.effective_mode,
		'llm_call_count': result.llm_call_count,
	}.items():
		ElementTree.SubElement(properties, 'property', {'name': str(key), 'value': str(value)})

	for item in result.step_results:
		testcase = ElementTree.SubElement(
			suite,
			'testcase',
			{'classname': 'web_ui_qa', 'name': item.step.step_id},
		)
		message = item.judgement.actual_result if item.judgement else item.status.value
		if item.status == QAStepStatus.SUT_FAILED:
			ElementTree.SubElement(testcase, 'failure', {'type': 'sut', 'message': message}).text = message
		elif item.status in {QAStepStatus.AGENT_FAILED, QAStepStatus.BLOCKED, QAStepStatus.INCONCLUSIVE}:
			ElementTree.SubElement(
				testcase,
				'error',
				{'type': item.status.value.casefold(), 'message': message},
			).text = message
		elif item.status == QAStepStatus.NOT_RUN:
			ElementTree.SubElement(testcase, 'skipped', {'message': 'Not run after an earlier terminal result'})
	if not has_step_results:
		testcase = ElementTree.SubElement(suite, 'testcase', {'classname': 'web_ui_qa', 'name': '__run__'})
		if failures:
			ElementTree.SubElement(testcase, 'failure', {'type': 'sut', 'message': result.summary}).text = result.summary
		elif errors:
			ElementTree.SubElement(
				testcase,
				'error',
				{'type': result.status.value.casefold(), 'message': result.summary},
			).text = result.summary
	return ElementTree.tostring(suite, encoding='unicode', xml_declaration=True)


def write_qa_junit_report(result: QARunResult, path: str | Path) -> Path:
	"""Write a CI-consumable JUnit XML report."""

	return _write_text(path, qa_junit_xml(result))


def qa_html_report(result: QARunResult) -> str:
	"""Render a self-contained, escaped evidence and judgement report."""

	def esc(value: object) -> str:
		return html.escape(str(value), quote=True)

	rows: list[str] = []
	for item in result.step_results:
		judgement = item.judgement
		receipt = item.evidence.action_receipt if item.evidence else None
		review = item.review
		evidence_links = []
		state_transition = '—'
		if item.evidence:
			state_transition = (
				f'BEFORE {item.evidence.before.url}\n{item.evidence.before.dom_summary}\n\n'
				f'AFTER {item.evidence.after.url}\n{item.evidence.after.dom_summary}'
			)
			for artifact in item.evidence.artifacts:
				label = f'{artifact.evidence_id} · {artifact.kind.value}: {artifact.summary}'
				if artifact.artifact_path:
					artifact_path = Path(artifact.artifact_path).expanduser().resolve()
					evidence_links.append(f'<a href="{esc(artifact_path.as_uri())}">{esc(label)}</a>')
				else:
					evidence_links.append(esc(label))
		rows.append(
			'<tr>'
			f'<td>{esc(item.step.step_id)}</td>'
			f'<td>{esc(item.step.instruction)}</td>'
			f'<td>{esc(item.step.expected_result)}</td>'
			f'<td><span class="status {esc(item.status.value)}">{esc(item.status.value)}</span></td>'
			f'<td>{esc(judgement.actual_result if judgement else "—")}</td>'
			f'<td><pre>{esc(state_transition)}</pre></td>'
			f'<td><pre>{esc(receipt.model_dump_json(indent=2) if receipt else "—")}</pre></td>'
			f'<td><pre>{esc(review.model_dump_json(indent=2) if review else "—")}</pre></td>'
			f'<td>{"<br>".join(evidence_links) or "—"}</td>'
			'</tr>'
		)
	cleanup_rows = (
		''.join(
			f'<li>{esc(item.step.step_id)} — {esc(item.status.value)} — {esc(item.reason)}</li>'
			for item in result.cleanup_results
		)
		or '<li>None</li>'
	)
	preconditions = (
		''.join(
			f'<li>{esc(item.precondition.precondition_id)} — {esc(item.status.value)} — {esc(item.reason)}</li>'
			for item in result.precondition_results
		)
		or '<li>None</li>'
	)
	warnings = ''.join(f'<li>{esc(value)}</li>' for value in result.warnings) or '<li>None</li>'
	return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>AutoTester Use QA Report</title>
<style>
body{{font:14px system-ui,sans-serif;margin:24px;color:#17202a}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #d8dee4;padding:8px;vertical-align:top;text-align:left}} th{{background:#f6f8fa}}
pre{{white-space:pre-wrap;max-width:34rem}} .status{{font-weight:700}} .PASSED{{color:#08783e}}
.SUT_FAILED{{color:#b42318}} .AGENT_FAILED,.BLOCKED,.INCONCLUSIVE{{color:#9a6700}} .NOT_RUN{{color:#656d76}}
</style></head><body>
<h1>AutoTester Use Web UI QA</h1>
<p><strong>Status:</strong> {esc(result.status.value)} · <strong>Run:</strong> {esc(result.run_id)} ·
<strong>Mode:</strong> {esc(result.requested_mode)} → {esc(result.effective_mode)} ·
<strong>LLM calls:</strong> {result.llm_call_count}</p>
<p>{esc(result.summary)}</p>
<h2>Preconditions</h2><ul>{preconditions}</ul>
<h2>Business steps</h2><table><thead><tr><th>ID</th><th>Operation</th><th>Expected</th><th>Status</th>
<th>Actual</th><th>Before / after</th><th>Action receipt</th><th>Review</th><th>Evidence</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Cleanup</h2><ul>{cleanup_rows}</ul><h2>Warnings</h2><ul>{warnings}</ul>
</body></html>"""


def write_qa_html_report(result: QARunResult, path: str | Path) -> Path:
	"""Write an escaped, self-contained HTML QA report."""

	return _write_text(path, qa_html_report(result))
