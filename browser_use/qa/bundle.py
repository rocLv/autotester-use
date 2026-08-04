"""Versioned local persistence for QA specifications, evidence, and replay traces."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from browser_use.qa.views import QARunResult, WebUITestCase


def qa_content_hash(value: str | None) -> str:
	"""Return a stable SHA-256 hash for cache and bundle identity checks."""

	return hashlib.sha256((value or '').strip().encode('utf-8')).hexdigest()


def _file_hash(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class QABundleRevision(BaseModel):
	"""One immutable revision inside a local QA bundle."""

	model_config = ConfigDict(extra='forbid')

	revision_id: str
	created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
	status: str
	files: dict[str, str]


class QABundleManifest(BaseModel):
	"""Integrity and compatibility metadata for a QA bundle directory."""

	model_config = ConfigDict(extra='forbid')

	schema_version: Literal[2] = 2
	task_hash: str
	ground_truth_hash: str
	root_url: str
	registrable_domain: str
	environment: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
	revisions: list[QABundleRevision] = Field(default_factory=list)


class QABundle(BaseModel):
	"""Loaded QA bundle ready for cross-process deterministic replay."""

	model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

	path: Path
	manifest: QABundleManifest
	revision: QABundleRevision
	test_case: WebUITestCase
	run_result: QARunResult
	action_history: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

	@classmethod
	def load(cls, path: str | Path, *, revision_id: str | None = None) -> QABundle:
		"""Load and verify a complete v2 bundle revision before browser execution."""

		bundle_path = Path(path).expanduser().resolve()
		manifest_path = bundle_path / 'manifest.json'
		if not manifest_path.is_file():
			raise ValueError(f'QA bundle manifest does not exist: {manifest_path}')
		manifest = QABundleManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
		if not manifest.revisions:
			raise ValueError('QA bundle contains no revisions')
		if revision_id is None:
			revision = manifest.revisions[-1]
		else:
			revision = next((item for item in manifest.revisions if item.revision_id == revision_id), None)
			if revision is None:
				raise ValueError(f'QA bundle revision does not exist: {revision_id}')

		for relative_path, expected_hash in revision.files.items():
			file_path = bundle_path / relative_path
			if not file_path.is_file():
				raise ValueError(f'QA bundle file is missing: {relative_path}')
			if _file_hash(file_path) != expected_hash:
				raise ValueError(f'QA bundle integrity check failed: {relative_path}')

		revision_root = bundle_path / 'revisions' / revision.revision_id
		test_case = WebUITestCase.model_validate_json((revision_root / 'test_case.json').read_text(encoding='utf-8'))
		run_payload = json.loads((revision_root / 'run_result.json').read_text(encoding='utf-8'))

		def resolve_artifact_paths(value: Any) -> Any:
			if isinstance(value, dict):
				updated = {key: resolve_artifact_paths(item) for key, item in value.items()}
				artifact_path = updated.get('artifact_path')
				if isinstance(artifact_path, str) and artifact_path:
					updated['artifact_path'] = str((revision_root / artifact_path).resolve())
				return updated
			if isinstance(value, list):
				return [resolve_artifact_paths(item) for item in value]
			return value

		run_result = QARunResult.model_validate(resolve_artifact_paths(run_payload))
		action_history: dict[str, list[dict[str, Any]]] = {}
		actions_root = revision_root / 'actions'
		if actions_root.is_dir():
			for actions_path in sorted(actions_root.glob('*.json')):
				payload = json.loads(actions_path.read_text(encoding='utf-8'))
				action_history[str(payload['step_id'])] = list(payload.get('history', []))
		else:
			# Compatibility with early pre-release v2 bundles.
			legacy_actions_path = revision_root / 'actions.json'
			if legacy_actions_path.exists():
				action_history = json.loads(legacy_actions_path.read_text(encoding='utf-8'))
		return cls(
			path=bundle_path,
			manifest=manifest,
			revision=revision,
			test_case=test_case,
			run_result=run_result,
			action_history=action_history,
		)

	@classmethod
	def save(
		cls,
		path: str | Path,
		*,
		task: str,
		ground_truth: str | None,
		run_result: QARunResult,
		action_history: dict[str, list[dict[str, Any]]],
	) -> QABundle:
		"""Append an immutable, checksummed revision and atomically publish its manifest."""

		if run_result.test_case is None:
			raise ValueError('A QA bundle requires a compiled test case')
		bundle_path = Path(path).expanduser().resolve()
		bundle_path.mkdir(parents=True, exist_ok=True)
		manifest_path = bundle_path / 'manifest.json'
		if manifest_path.exists():
			manifest = QABundleManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
			if manifest.task_hash != qa_content_hash(task):
				raise ValueError('Existing QA bundle belongs to a different Task')
			if manifest.ground_truth_hash != qa_content_hash(ground_truth):
				raise ValueError('Existing QA bundle belongs to different ground_truth')
			if manifest.root_url != run_result.test_case.root_url:
				raise ValueError('Existing QA bundle belongs to a different root URL')
		else:
			manifest = QABundleManifest(
				task_hash=qa_content_hash(task),
				ground_truth_hash=qa_content_hash(ground_truth),
				root_url=run_result.test_case.root_url,
				registrable_domain=run_result.test_case.registrable_domain,
				environment=run_result.environment,
			)
		manifest.environment = run_result.environment

		revision_id = f'{datetime.now(UTC).strftime("%Y%m%dT%H%M%S")}_{uuid4().hex[:10]}'
		revisions_root = bundle_path / 'revisions'
		revisions_root.mkdir(parents=True, exist_ok=True)
		temporary_root = Path(tempfile.mkdtemp(prefix=f'.{revision_id}_', dir=revisions_root))
		try:
			_write_json(temporary_root / 'test_case.json', run_result.test_case.model_dump(mode='json'))
			used_action_names: set[str] = set()
			for step_id, history in action_history.items():
				safe_step_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', step_id).strip('._') or 'step'
				if safe_step_id in used_action_names:
					safe_step_id = f'{safe_step_id}_{qa_content_hash(step_id)[:10]}'
				used_action_names.add(safe_step_id)
				_write_json(
					temporary_root / 'actions' / f'{safe_step_id}.json',
					{'step_id': step_id, 'history': history},
				)
			run_payload = run_result.model_dump(mode='json')
			artifact_relative_paths: dict[str, str] = {}
			for artifact in run_result.artifacts:
				if not artifact.redacted or not artifact.artifact_path:
					continue
				source = Path(artifact.artifact_path)
				if not source.is_file():
					continue
				suffix = ''.join(source.suffixes) or '.bin'
				relative = Path('artifacts') / f'{artifact.evidence_id}{suffix}'
				destination = temporary_root / relative
				destination.parent.mkdir(parents=True, exist_ok=True)
				shutil.copy2(source, destination)
				artifact_relative_paths[artifact.evidence_id] = str(relative)

			def rewrite_artifact_paths(value: Any) -> Any:
				if isinstance(value, dict):
					updated = {key: rewrite_artifact_paths(item) for key, item in value.items()}
					evidence_id = updated.get('evidence_id')
					if isinstance(evidence_id, str):
						updated['artifact_path'] = artifact_relative_paths.get(evidence_id)
					return updated
				if isinstance(value, list):
					return [rewrite_artifact_paths(item) for item in value]
				return value

			_write_json(temporary_root / 'run_result.json', rewrite_artifact_paths(run_payload))
			final_root = revisions_root / revision_id
			os.replace(temporary_root, final_root)
		except Exception:
			shutil.rmtree(temporary_root, ignore_errors=True)
			raise

		files = {str(file.relative_to(bundle_path)): _file_hash(file) for file in final_root.rglob('*') if file.is_file()}
		revision = QABundleRevision(revision_id=revision_id, status=run_result.status.value, files=files)
		manifest.revisions.append(revision)
		manifest_temp = bundle_path / f'.manifest.{uuid4().hex}.tmp'
		_write_json(manifest_temp, manifest.model_dump(mode='json'))
		os.replace(manifest_temp, manifest_path)
		return cls.load(bundle_path, revision_id=revision_id)
