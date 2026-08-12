"""Chrome DevTools Recorder import, replay, and export helpers."""

from __future__ import annotations

import asyncio
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
	from browser_use.agent.views import AgentHistoryList
	from browser_use.browser.session import BrowserSession


class ChromeRecorderStepStatus(StrEnum):
	"""Execution status for one Chrome Recorder step."""

	PASSED = 'PASSED'
	FAILED = 'FAILED'
	SKIPPED = 'SKIPPED'


class ChromeRecorderPlaybackStatus(StrEnum):
	"""Terminal status for a Chrome Recorder replay."""

	PASSED = 'PASSED'
	FAILED = 'FAILED'
	PARTIAL = 'PARTIAL'


class ChromeRecorderStep(BaseModel):
	"""One step from a Chrome DevTools Recorder JSON flow.

	Chrome's schema changes over time, so unknown fields are preserved.
	"""

	model_config = ConfigDict(extra='allow')

	type: str = Field(min_length=1)
	url: str | None = None
	target: str | None = None
	selectors: list[list[str] | str] = Field(default_factory=list)
	value: str | None = None
	key: str | None = None
	expression: str | None = None
	timeout: int | float | None = None
	duration: int | float | None = None
	x: int | float | None = None
	y: int | float | None = None
	offsetX: int | float | None = None
	offsetY: int | float | None = None
	button: str | None = None
	width: int | None = None
	height: int | None = None
	deviceScaleFactor: float | None = None
	isMobile: bool | None = None
	hasTouch: bool | None = None
	isLandscape: bool | None = None

	@field_validator('type')
	@classmethod
	def strip_type(cls, value: str) -> str:
		value = value.strip()
		if not value:
			raise ValueError('Chrome Recorder step type must not be blank')
		return value

	def selector_groups(self) -> list[list[str]]:
		"""Return selectors as Chrome-style ordered fallback groups."""

		groups: list[list[str]] = []
		for group in self.selectors:
			if isinstance(group, str):
				candidates = [group]
			else:
				candidates = group
			cleaned = [candidate.strip() for candidate in candidates if isinstance(candidate, str) and candidate.strip()]
			if cleaned:
				groups.append(cleaned)
		return groups

	def timeout_ms(self, default: int) -> int:
		"""Return the step timeout in milliseconds."""

		for value in (self.timeout, self.duration):
			if value is not None:
				return max(0, int(value))
		return default


class ChromeRecorderFlow(BaseModel):
	"""A Chrome DevTools Recorder flow."""

	model_config = ConfigDict(extra='allow')

	title: str = Field(default='Chrome recording', min_length=1)
	steps: list[ChromeRecorderStep] = Field(default_factory=list)
	timeout: int | float | None = None
	selectorAttribute: str | None = None

	@field_validator('title')
	@classmethod
	def strip_title(cls, value: str) -> str:
		value = value.strip()
		if not value:
			raise ValueError('Chrome Recorder flow title must not be blank')
		return value


class ChromeRecorderStepResult(BaseModel):
	"""Replay result for one Chrome Recorder step."""

	model_config = ConfigDict(extra='forbid')

	index: int = Field(ge=1)
	type: str
	status: ChromeRecorderStepStatus
	message: str = ''
	url: str | None = None
	selector: str | None = None
	details: dict[str, Any] = Field(default_factory=dict)


class ChromeRecorderPlaybackResult(BaseModel):
	"""Replay result for an entire Chrome Recorder flow."""

	model_config = ConfigDict(extra='forbid')

	title: str
	status: ChromeRecorderPlaybackStatus
	steps: list[ChromeRecorderStepResult]

	@property
	def is_successful(self) -> bool:
		return self.status == ChromeRecorderPlaybackStatus.PASSED


class ChromeRecorderPlaybackError(RuntimeError):
	"""Raised when a Chrome Recorder step cannot be replayed."""


def load_chrome_recorder_flow(source: str | Path | Mapping[str, Any] | ChromeRecorderFlow) -> ChromeRecorderFlow:
	"""Load a Chrome DevTools Recorder flow from a path, JSON string, mapping, or model."""

	if isinstance(source, ChromeRecorderFlow):
		return source
	if isinstance(source, Mapping):
		return ChromeRecorderFlow.model_validate(source)
	if isinstance(source, Path):
		return ChromeRecorderFlow.model_validate_json(source.read_text(encoding='utf-8'))
	if isinstance(source, str):
		trimmed = source.strip()
		if trimmed.startswith('{'):
			return ChromeRecorderFlow.model_validate_json(trimmed)
		return ChromeRecorderFlow.model_validate_json(Path(source).read_text(encoding='utf-8'))
	raise TypeError(f'Unsupported Chrome Recorder flow source: {type(source).__name__}')


def write_chrome_recorder_flow(flow: ChromeRecorderFlow | Mapping[str, Any], path: str | Path) -> Path:
	"""Write a Chrome DevTools Recorder flow JSON file."""

	resolved_flow = flow if isinstance(flow, ChromeRecorderFlow) else ChromeRecorderFlow.model_validate(flow)
	resolved_path = Path(path)
	resolved_path.parent.mkdir(parents=True, exist_ok=True)
	resolved_path.write_text(
		json.dumps(resolved_flow.model_dump(mode='json', exclude_none=True), ensure_ascii=False, indent=2) + '\n',
		encoding='utf-8',
	)
	return resolved_path


class ChromeRecorderPlayer:
	"""Replay Chrome DevTools Recorder flows against a BrowserSession."""

	def __init__(
		self,
		browser_session: BrowserSession,
		*,
		default_timeout_ms: int = 5000,
		stop_on_error: bool = True,
		skip_unsupported: bool = False,
		wait_after_step_ms: int = 0,
	) -> None:
		self.browser_session = browser_session
		self.default_timeout_ms = default_timeout_ms
		self.stop_on_error = stop_on_error
		self.skip_unsupported = skip_unsupported
		self.wait_after_step_ms = wait_after_step_ms

	async def replay(
		self,
		flow: str | Path | Mapping[str, Any] | ChromeRecorderFlow,
		*,
		max_steps: int | None = None,
	) -> ChromeRecorderPlaybackResult:
		"""Replay a Chrome DevTools Recorder flow."""

		resolved_flow = load_chrome_recorder_flow(flow)
		results: list[ChromeRecorderStepResult] = []
		steps = resolved_flow.steps[:max_steps] if max_steps is not None else resolved_flow.steps

		for index, step in enumerate(steps, start=1):
			try:
				result = await self._replay_step(index, step, resolved_flow)
			except Exception as exc:
				result = ChromeRecorderStepResult(
					index=index,
					type=step.type,
					status=ChromeRecorderStepStatus.FAILED,
					message=str(exc),
				)
			results.append(result)
			if self.wait_after_step_ms > 0:
				await asyncio.sleep(self.wait_after_step_ms / 1000)
			if result.status == ChromeRecorderStepStatus.FAILED and self.stop_on_error:
				break

		status = self._playback_status(results, total_steps=len(steps))
		return ChromeRecorderPlaybackResult(title=resolved_flow.title, status=status, steps=results)

	async def _replay_step(
		self,
		index: int,
		step: ChromeRecorderStep,
		flow: ChromeRecorderFlow,
	) -> ChromeRecorderStepResult:
		step_type = step.type

		if step_type == 'navigate':
			if not step.url:
				raise ChromeRecorderPlaybackError('navigate step is missing url')
			await self.browser_session.navigate_to(step.url, new_tab=False)
			return ChromeRecorderStepResult(
				index=index,
				type=step_type,
				status=ChromeRecorderStepStatus.PASSED,
				message=f'Navigated to {step.url}',
				url=step.url,
			)

		if step_type == 'setViewport':
			await self._set_viewport(step)
			return ChromeRecorderStepResult(
				index=index,
				type=step_type,
				status=ChromeRecorderStepStatus.PASSED,
				message='Viewport updated',
				details={'width': step.width, 'height': step.height},
			)

		if step_type in {'click', 'doubleClick'}:
			if step.selector_groups():
				return await self._run_dom_action(index, step, 'click', click_count=2 if step_type == 'doubleClick' else 1)
			if step.x is not None and step.y is not None:
				await self._click_coordinates(step)
				return ChromeRecorderStepResult(
					index=index,
					type=step_type,
					status=ChromeRecorderStepStatus.PASSED,
					message=f'Clicked coordinates ({int(step.x)}, {int(step.y)})',
				)
			raise ChromeRecorderPlaybackError('click step has no selectors or coordinates')

		if step_type in {'change', 'input'}:
			return await self._run_dom_action(index, step, 'change', value=step.value or '')

		if step_type == 'hover':
			return await self._run_dom_action(index, step, 'hover')

		if step_type in {'scroll', 'wheel'}:
			return await self._scroll(index, step)

		if step_type == 'waitForElement':
			return await self._run_dom_action(
				index, step, 'waitForElement', timeout_ms=step.timeout_ms(self._flow_timeout_ms(flow))
			)

		if step_type == 'waitForExpression':
			await self._wait_for_expression(step.expression, timeout_ms=step.timeout_ms(self._flow_timeout_ms(flow)))
			return ChromeRecorderStepResult(
				index=index,
				type=step_type,
				status=ChromeRecorderStepStatus.PASSED,
				message='Expression returned truthy',
			)

		if step_type == 'waitForTimeout':
			timeout_ms = step.timeout_ms(self.default_timeout_ms)
			await asyncio.sleep(timeout_ms / 1000)
			return ChromeRecorderStepResult(
				index=index,
				type=step_type,
				status=ChromeRecorderStepStatus.PASSED,
				message=f'Waited {timeout_ms}ms',
			)

		if step_type in {'keyDown', 'keyUp'}:
			if not step.key:
				raise ChromeRecorderPlaybackError(f'{step_type} step is missing key')
			await self._dispatch_key(step_type, step.key)
			return ChromeRecorderStepResult(
				index=index,
				type=step_type,
				status=ChromeRecorderStepStatus.PASSED,
				message=f'Dispatched {step_type} {step.key}',
			)

		if self.skip_unsupported:
			return ChromeRecorderStepResult(
				index=index,
				type=step_type,
				status=ChromeRecorderStepStatus.SKIPPED,
				message=f'Unsupported Chrome Recorder step type: {step_type}',
			)
		raise ChromeRecorderPlaybackError(f'Unsupported Chrome Recorder step type: {step_type}')

	async def _run_dom_action(
		self, index: int, step: ChromeRecorderStep, action: str, **options: Any
	) -> ChromeRecorderStepResult:
		payload = {
			'action': action,
			'selectors': step.selector_groups(),
			'value': options.get('value'),
			'offsetX': step.offsetX,
			'offsetY': step.offsetY,
			'button': step.button or 'left',
			'clickCount': options.get('click_count', 1),
			'timeoutMs': options.get('timeout_ms', self.default_timeout_ms),
		}
		result = await self._evaluate_json(_dom_action_expression(payload))
		if not isinstance(result, dict):
			raise ChromeRecorderPlaybackError(f'{action} returned a non-object result')
		if not result.get('ok'):
			reason = result.get('reason') or 'dom_action_failed'
			tried = result.get('tried') or []
			selector = tried[0] if tried else None
			return ChromeRecorderStepResult(
				index=index,
				type=step.type,
				status=ChromeRecorderStepStatus.FAILED,
				message=str(reason),
				selector=selector,
				details=result,
			)
		return ChromeRecorderStepResult(
			index=index,
			type=step.type,
			status=ChromeRecorderStepStatus.PASSED,
			message=str(result.get('summary') or f'{action} completed'),
			selector=result.get('selector'),
			details=result,
		)

	async def _scroll(self, index: int, step: ChromeRecorderStep) -> ChromeRecorderStepResult:
		payload = {
			'selectors': step.selector_groups(),
			'x': step.x,
			'y': step.y,
			'offsetX': step.offsetX,
			'offsetY': step.offsetY,
		}
		result = await self._evaluate_json(_scroll_expression(payload))
		if not isinstance(result, dict) or not result.get('ok'):
			raise ChromeRecorderPlaybackError(str(result.get('reason') if isinstance(result, dict) else 'scroll failed'))
		return ChromeRecorderStepResult(
			index=index,
			type=step.type,
			status=ChromeRecorderStepStatus.PASSED,
			message=str(result.get('summary') or 'Scrolled'),
			selector=result.get('selector'),
			details=result,
		)

	async def _wait_for_expression(self, expression: str | None, *, timeout_ms: int) -> None:
		if not expression:
			raise ChromeRecorderPlaybackError('waitForExpression step is missing expression')
		payload = {'expression': expression, 'timeoutMs': timeout_ms}
		result = await self._evaluate_json(_wait_for_expression(payload))
		if not isinstance(result, dict) or not result.get('ok'):
			raise ChromeRecorderPlaybackError(str(result.get('reason') if isinstance(result, dict) else 'expression failed'))

	async def _set_viewport(self, step: ChromeRecorderStep) -> None:
		if step.width is None or step.height is None:
			raise ChromeRecorderPlaybackError('setViewport step is missing width or height')
		cdp_session = await self.browser_session.get_or_create_cdp_session()
		params = {
			'width': int(step.width),
			'height': int(step.height),
			'deviceScaleFactor': float(step.deviceScaleFactor if step.deviceScaleFactor is not None else 1),
			'mobile': bool(step.isMobile),
		}
		await cdp_session.cdp_client.send.Emulation.setDeviceMetricsOverride(
			params=params,
			session_id=cdp_session.session_id,
		)

	async def _click_coordinates(self, step: ChromeRecorderStep) -> None:
		x = int(step.x or 0)
		y = int(step.y or 0)
		button = _normalize_mouse_button(step.button)
		cdp_session = await self.browser_session.get_or_create_cdp_session()
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={'type': 'mousePressed', 'x': x, 'y': y, 'button': button, 'clickCount': 1},
			session_id=cdp_session.session_id,
		)
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={'type': 'mouseReleased', 'x': x, 'y': y, 'button': button, 'clickCount': 1},
			session_id=cdp_session.session_id,
		)

	async def _dispatch_key(self, step_type: str, key: str) -> None:
		cdp_session = await self.browser_session.get_or_create_cdp_session()
		params: dict[str, Any] = {'type': step_type, 'key': key}
		if len(key) == 1 and step_type == 'keyDown':
			params['text'] = key
		await cdp_session.cdp_client.send.Input.dispatchKeyEvent(params=params, session_id=cdp_session.session_id)

	async def _evaluate_json(self, expression: str) -> Any:
		cdp_session = await self.browser_session.get_or_create_cdp_session()
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={'expression': expression, 'awaitPromise': True, 'returnByValue': True},
			session_id=cdp_session.session_id,
		)
		if result.get('exceptionDetails'):
			raise ChromeRecorderPlaybackError(str(result['exceptionDetails']))
		remote_result = result.get('result') or {}
		if 'value' in remote_result:
			return remote_result['value']
		return remote_result.get('description')

	def _flow_timeout_ms(self, flow: ChromeRecorderFlow) -> int:
		if flow.timeout is None:
			return self.default_timeout_ms
		return max(0, int(flow.timeout))

	@staticmethod
	def _playback_status(
		results: list[ChromeRecorderStepResult],
		*,
		total_steps: int,
	) -> ChromeRecorderPlaybackStatus:
		if any(result.status == ChromeRecorderStepStatus.FAILED for result in results):
			return ChromeRecorderPlaybackStatus.FAILED
		if len(results) < total_steps or any(result.status == ChromeRecorderStepStatus.SKIPPED for result in results):
			return ChromeRecorderPlaybackStatus.PARTIAL
		return ChromeRecorderPlaybackStatus.PASSED


def export_agent_history_to_chrome_recorder(
	history: AgentHistoryList[Any] | list[dict[str, Any]] | list[list[dict[str, Any]]],
	*,
	title: str = 'Browser Use recording',
) -> ChromeRecorderFlow:
	"""Convert Browser Use action history into a Chrome DevTools Recorder flow."""

	steps: list[ChromeRecorderStep] = []
	for action in _iter_action_dicts(history):
		steps.extend(_action_to_chrome_steps(action))
	return ChromeRecorderFlow(title=title, steps=steps)


def write_agent_history_chrome_recorder_flow(
	history: AgentHistoryList[Any] | list[dict[str, Any]] | list[list[dict[str, Any]]],
	path: str | Path,
	*,
	title: str = 'Browser Use recording',
) -> Path:
	"""Write Browser Use action history as a Chrome DevTools Recorder JSON flow."""

	return write_chrome_recorder_flow(export_agent_history_to_chrome_recorder(history, title=title), path)


def _iter_action_dicts(history: Any) -> list[dict[str, Any]]:
	if hasattr(history, 'model_actions'):
		return list(history.model_actions())
	if hasattr(history, 'action_history'):
		return [action for step in history.action_history() for action in step]
	if isinstance(history, list):
		if all(isinstance(item, list) for item in history):
			return [action for step in history for action in step if isinstance(action, dict)]
		return [action for action in history if isinstance(action, dict)]
	raise TypeError(f'Unsupported action history source: {type(history).__name__}')


def _action_to_chrome_steps(action: dict[str, Any]) -> list[ChromeRecorderStep]:
	action_name, params = _extract_action_payload(action)
	if action_name is None:
		return []
	params = params if isinstance(params, dict) else {}
	interacted_element = action.get('interacted_element')
	selectors = _selectors_for_interacted_element(interacted_element)

	if action_name == 'navigate':
		url = params.get('url')
		return [ChromeRecorderStep(type='navigate', url=url)] if isinstance(url, str) and url else []

	if action_name == 'click':
		if params.get('coordinate_x') is not None and params.get('coordinate_y') is not None:
			return [
				ChromeRecorderStep(
					type='click',
					x=params.get('coordinate_x'),
					y=params.get('coordinate_y'),
				)
			]
		return [ChromeRecorderStep(type='click', selectors=selectors)] if selectors else []

	if action_name == 'input':
		text = params.get('text')
		return [ChromeRecorderStep(type='change', selectors=selectors, value=str(text or ''))] if selectors else []

	if action_name == 'wait':
		seconds = params.get('seconds', 3)
		try:
			timeout = int(float(seconds) * 1000)
		except (TypeError, ValueError):
			timeout = 3000
		return [ChromeRecorderStep(type='waitForTimeout', timeout=timeout)]

	if action_name == 'scroll':
		pages = params.get('pages', 1)
		try:
			y = int(float(pages) * 700)
		except (TypeError, ValueError):
			y = 700
		if params.get('down') is False:
			y = -y
		return [ChromeRecorderStep(type='scroll', y=y)]

	if action_name == 'send_keys':
		keys = str(params.get('keys') or '').strip()
		return _key_steps(keys)

	return []


def _extract_action_payload(action: dict[str, Any]) -> tuple[str | None, Any]:
	for key, value in action.items():
		if key not in {'interacted_element', 'result'}:
			return key, value
	return None, None


def _key_steps(keys: str) -> list[ChromeRecorderStep]:
	if not keys:
		return []
	steps: list[ChromeRecorderStep] = []
	for key in keys.split():
		steps.append(ChromeRecorderStep(type='keyDown', key=key))
		steps.append(ChromeRecorderStep(type='keyUp', key=key))
	return steps


def _selectors_for_interacted_element(element: Any) -> list[list[str]]:
	if element is None:
		return []
	attrs = _element_value(element, 'attributes') or {}
	if not isinstance(attrs, dict):
		attrs = {}

	selectors: list[list[str]] = []
	element_id = attrs.get('id')
	if isinstance(element_id, str) and element_id:
		selectors.append([_css_id_selector(element_id)])
	aria_name = _element_value(element, 'ax_name') or attrs.get('aria-label') or attrs.get('title') or attrs.get('placeholder')
	if isinstance(aria_name, str) and aria_name.strip():
		selectors.append([f'aria/{aria_name.strip()}'])
	name = attrs.get('name')
	if isinstance(name, str) and name:
		selectors.append([f'css/[name="{_css_attr_escape(name)}"]'])
	xpath = _element_value(element, 'x_path')
	if isinstance(xpath, str) and xpath:
		selectors.append([f'xpath/{xpath}'])
	return selectors


def _element_value(element: Any, key: str) -> Any:
	if isinstance(element, dict):
		return element.get(key)
	return getattr(element, key, None)


_CSS_IDENT_RE = re.compile(r'^-?[_a-zA-Z][-_a-zA-Z0-9]*$')


def _css_id_selector(value: str) -> str:
	if _CSS_IDENT_RE.match(value):
		return f'css/#{value}'
	return f'css/[id="{_css_attr_escape(value)}"]'


def _css_attr_escape(value: str) -> str:
	return value.replace('\\', '\\\\').replace('"', '\\"')


def _normalize_mouse_button(button: str | None) -> str:
	if button in {'left', 'right', 'middle'}:
		return button
	return 'left'


def _dom_action_expression(payload: dict[str, Any]) -> str:
	return f"""
(() => {{
	const payload = {json.dumps(payload)};
	const helpers = {_recorder_dom_helpers()};
	return helpers.runDomAction(payload);
}})()
"""


def _scroll_expression(payload: dict[str, Any]) -> str:
	return f"""
(() => {{
	const payload = {json.dumps(payload)};
	const helpers = {_recorder_dom_helpers()};
	return helpers.runScroll(payload);
}})()
"""


def _wait_for_expression(payload: dict[str, Any]) -> str:
	return f"""
(() => {{
	const payload = {json.dumps(payload)};
	return new Promise((resolve) => {{
		const deadline = Date.now() + Math.max(0, Number(payload.timeoutMs || 0));
		const check = () => {{
			try {{
				if (Boolean(Function('"use strict"; return (' + payload.expression + ')')())) {{
					resolve({{ok: true, summary: 'Expression returned truthy'}});
					return;
				}}
			}} catch (error) {{
				resolve({{ok: false, reason: String(error && error.message ? error.message : error)}});
				return;
			}}
			if (Date.now() >= deadline) {{
				resolve({{ok: false, reason: 'expression_timeout'}});
				return;
			}}
			setTimeout(check, 100);
		}};
		check();
	}});
}})()
"""


def _recorder_dom_helpers() -> str:
	return r"""
(() => {
	const normalizeGroups = (selectors) => {
		if (!Array.isArray(selectors)) return [];
		return selectors
			.map((group) => Array.isArray(group) ? group : [group])
			.map((group) => group.filter((selector) => typeof selector === 'string' && selector.trim()).map((selector) => selector.trim()))
			.filter((group) => group.length);
	};

	const textOf = (node) => String(node?.innerText || node?.textContent || '').replace(/\s+/g, ' ').trim();

	const xpath = (selector) => {
		const path = selector.startsWith('xpath/') ? selector.slice(6) : selector;
		return document.evaluate(path, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
	};

	const queryPierce = (selector, root = document) => {
		const found = root.querySelector(selector);
		if (found) return found;
		for (const child of root.querySelectorAll('*')) {
			if (child.shadowRoot) {
				const shadowFound = queryPierce(selector, child.shadowRoot);
				if (shadowFound) return shadowFound;
			}
		}
		return null;
	};

	const accessibleName = (node) => {
		const labelledBy = node.getAttribute?.('aria-labelledby');
		const labelledText = labelledBy
			? labelledBy.split(/\s+/).map((id) => textOf(document.getElementById(id))).filter(Boolean).join(' ')
			: '';
		return [
			labelledText,
			node.getAttribute?.('aria-label'),
			node.getAttribute?.('alt'),
			node.getAttribute?.('title'),
			node.getAttribute?.('placeholder'),
			node.value,
			textOf(node),
		].filter(Boolean).join(' ').replace(/\s+/g, ' ').trim();
	};

	const aria = (selector) => {
		const raw = selector.slice(5).replace(/\[.*\]$/, '').replace(/^"|"$/g, '').trim();
		const expected = raw.toLowerCase();
		for (const node of document.querySelectorAll('button, a, input, textarea, select, [role], [aria-label], [aria-labelledby], [title]')) {
			const name = accessibleName(node).toLowerCase();
			if (name === expected || name.includes(expected)) return node;
		}
		return null;
	};

	const text = (selector) => {
		const raw = selector.slice(5).replace(/^"|"$/g, '').trim().toLowerCase();
		for (const node of document.querySelectorAll('body *')) {
			const value = textOf(node).toLowerCase();
			if (value === raw || value.includes(raw)) return node;
		}
		return null;
	};

	const findOne = (selector) => {
		if (selector.startsWith('css/')) return document.querySelector(selector.slice(4));
		if (selector.startsWith('xpath/')) return xpath(selector);
		if (selector.startsWith('aria/')) return aria(selector);
		if (selector.startsWith('text/')) return text(selector);
		if (selector.startsWith('pierce/')) return queryPierce(selector.slice(7));
		return document.querySelector(selector);
	};

	const findElement = (selectors) => {
		const tried = [];
		for (const group of normalizeGroups(selectors)) {
			for (const selector of group) {
				tried.push(selector);
				try {
					const element = findOne(selector);
					if (element) return {element, selector, tried};
				} catch (error) {
					tried.push(`${selector}: ${String(error && error.message ? error.message : error)}`);
				}
			}
		}
		return {element: null, selector: null, tried};
	};

	const setNativeValue = (element, value) => {
		if (element.isContentEditable) {
			element.textContent = String(value);
		} else if (element.tagName === 'SELECT') {
			element.value = String(value);
		} else {
			const prototype = element.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
			const descriptor = Object.getOwnPropertyDescriptor(prototype, 'value');
			if (descriptor && descriptor.set) descriptor.set.call(element, String(value));
			else element.value = String(value);
		}
		element.dispatchEvent(new InputEvent('input', {bubbles: true, inputType: 'insertText', data: String(value)}));
		element.dispatchEvent(new Event('change', {bubbles: true}));
	};

	const buttonCode = (button) => ({left: 0, middle: 1, right: 2})[button] ?? 0;

	const runDomAction = (payload) => new Promise((resolve) => {
		const deadline = Date.now() + Math.max(0, Number(payload.timeoutMs || 0));
		const attempt = () => {
			const match = findElement(payload.selectors);
			if (!match.element) {
				if (payload.action === 'waitForElement' && Date.now() < deadline) {
					setTimeout(attempt, 100);
					return;
				}
				resolve({ok: false, reason: 'element_not_found', tried: match.tried});
				return;
			}

			const element = match.element;
			element.scrollIntoView({block: 'center', inline: 'center'});
			element.focus?.({preventScroll: true});
			if (payload.action === 'waitForElement') {
				resolve({ok: true, summary: 'Element appeared', selector: match.selector});
				return;
			}
			if (payload.action === 'change') {
				setNativeValue(element, payload.value ?? '');
				resolve({ok: true, summary: 'Changed element value', selector: match.selector, value: element.value ?? textOf(element)});
				return;
			}
			if (payload.action === 'hover') {
				const rect = element.getBoundingClientRect();
				element.dispatchEvent(new MouseEvent('mousemove', {bubbles: true, cancelable: true, clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2}));
				resolve({ok: true, summary: 'Hovered element', selector: match.selector});
				return;
			}
			if (payload.action === 'click') {
				const rect = element.getBoundingClientRect();
				const clientX = rect.left + (Number.isFinite(Number(payload.offsetX)) ? Number(payload.offsetX) : rect.width / 2);
				const clientY = rect.top + (Number.isFinite(Number(payload.offsetY)) ? Number(payload.offsetY) : rect.height / 2);
				const options = {bubbles: true, cancelable: true, view: window, clientX, clientY, button: buttonCode(payload.button)};
				element.dispatchEvent(new PointerEvent('pointerdown', options));
				element.dispatchEvent(new MouseEvent('mousedown', options));
				for (let i = 0; i < Math.max(1, Number(payload.clickCount || 1)); i += 1) element.click();
				element.dispatchEvent(new MouseEvent('mouseup', options));
				element.dispatchEvent(new PointerEvent('pointerup', options));
				resolve({ok: true, summary: 'Clicked element', selector: match.selector, x: clientX, y: clientY});
				return;
			}
			resolve({ok: false, reason: `unsupported_dom_action:${payload.action}`, selector: match.selector});
		};
		attempt();
	});

	const runScroll = (payload) => {
		const x = Number(payload.x ?? 0);
		const y = Number(payload.y ?? 0);
		const match = normalizeGroups(payload.selectors).length ? findElement(payload.selectors) : {element: null, selector: null, tried: []};
		const target = match.element || window;
		if (target === window) window.scrollBy(x, y);
		else target.scrollBy(x, y);
		return {ok: true, summary: 'Scrolled', selector: match.selector, x, y};
	};

	return {runDomAction, runScroll};
})()
"""
