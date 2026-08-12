"""Objective console and network evidence collection for Web UI QA steps."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from browser_use.qa.navigation import NavigationScope


class QAEvidenceCursor(BaseModel):
	"""Offsets delimiting diagnostics produced by one business-step attempt."""

	network_index: int = Field(ge=0)
	console_index: int = Field(ge=0)
	network_event_index: int = Field(default=0, ge=0)


class QANetworkEvent(BaseModel):
	"""Redacted response metadata retained without response bodies or headers."""

	request_id: str
	method: str = 'GET'
	url: str
	resource_type: str = 'unknown'
	status: int | None = None
	error: str | None = None


class QADiagnostics(BaseModel):
	"""Relevant raw browser diagnostics captured since a cursor."""

	network_errors: list[str] = Field(default_factory=list)
	console_errors: list[str] = Field(default_factory=list)
	network_events: list[QANetworkEvent] = Field(default_factory=list)


def _get(value: Any, key: str, default: Any = None) -> Any:
	if isinstance(value, dict):
		return value.get(key, default)
	return getattr(value, key, default)


class QAEvidenceMonitor:
	"""Collect failed requests, HTTP errors, and unhandled page exceptions via CDP."""

	_MAX_ITEMS = 500
	_SUCCESS_EVIDENCE_RESOURCE_TYPES = frozenset({'Document', 'Fetch', 'XHR'})

	def __init__(self, browser_session: Any, navigation_scope: NavigationScope | None = None):
		self.browser_session = browser_session
		self.navigation_scope = navigation_scope
		self.network_errors: list[str] = []
		self.network_events: list[QANetworkEvent] = []
		self.console_errors: list[str] = []
		self._request_urls: dict[str, str] = {}
		self._request_methods: dict[str, str] = {}
		self._eligible_request_ids: set[str] = set()
		self._started = False

	async def start(self) -> None:
		"""Enable CDP domains and register read-only diagnostic callbacks once."""

		if self._started:
			return
		cdp_session = await self.browser_session.get_or_create_cdp_session()
		await cdp_session.cdp_client.send.Network.enable(session_id=cdp_session.session_id)
		await cdp_session.cdp_client.send.Runtime.enable(session_id=cdp_session.session_id)
		register = self.browser_session.cdp_client.register
		register.Network.requestWillBeSent(self._on_request)
		register.Network.responseReceived(self._on_response)
		register.Network.loadingFailed(self._on_loading_failed)
		register.Runtime.consoleAPICalled(self._on_console)
		register.Runtime.exceptionThrown(self._on_exception)
		self._started = True

	def cursor(self) -> QAEvidenceCursor:
		"""Return a stable cursor for the next step attempt."""

		return QAEvidenceCursor(
			network_index=len(self.network_errors),
			console_index=len(self.console_errors),
			network_event_index=len(self.network_events),
		)

	def since(self, cursor: QAEvidenceCursor | None) -> QADiagnostics:
		"""Return diagnostics captured after ``cursor`` without consuming them."""

		if cursor is None:
			return QADiagnostics()
		return QADiagnostics(
			network_errors=self.network_errors[cursor.network_index :],
			console_errors=self.console_errors[cursor.console_index :],
			network_events=self.network_events[cursor.network_event_index :],
		)

	def _append_network_event(self, event: QANetworkEvent) -> None:
		self.network_events.append(event)
		if len(self.network_events) > self._MAX_ITEMS:
			del self.network_events[: len(self.network_events) - self._MAX_ITEMS]

	def _append_network(self, message: str) -> None:
		self.network_errors.append(message[:2000])
		if len(self.network_errors) > self._MAX_ITEMS:
			del self.network_errors[: len(self.network_errors) - self._MAX_ITEMS]

	def _append_console(self, message: str) -> None:
		self.console_errors.append(message[:2000])
		if len(self.console_errors) > self._MAX_ITEMS:
			del self.console_errors[: len(self.console_errors) - self._MAX_ITEMS]

	def _on_request(self, params: Any, _session_id: str | None) -> None:
		request_id = _get(params, 'requestId')
		request = _get(params, 'request', {})
		url = _get(request, 'url')
		if request_id and url:
			request_id = str(request_id)
			document_url = str(_get(params, 'documentURL') or '')
			resource_type = str(_get(params, 'type') or '')
			is_qa_request = self.navigation_scope is None
			if self.navigation_scope is not None:
				context_url = str(url) if resource_type == 'Document' else document_url
				is_qa_request = bool(context_url and self.navigation_scope.allows(context_url))
			if is_qa_request:
				self._request_urls[request_id] = str(url)
				self._request_methods[request_id] = str(_get(request, 'method') or 'GET')
				self._eligible_request_ids.add(request_id)

	def _on_response(self, params: Any, _session_id: str | None) -> None:
		request_id = str(_get(params, 'requestId', ''))
		if self.navigation_scope is not None and request_id not in self._eligible_request_ids:
			return
		response = _get(params, 'response', {})
		status = _get(response, 'status')
		try:
			status_code = int(status)
		except (TypeError, ValueError):
			return
		url = str(_get(response, 'url') or self._request_urls.get(request_id, 'unknown URL'))
		resource_type = str(_get(params, 'type', 'unknown'))
		# Successful images, fonts, stylesheets, and data URLs are page-loading noise,
		# not business evidence. Keeping them can flood the Judge context with encoded
		# assets and prevent it from citing the small set of relevant evidence IDs.
		if status_code >= 400 or resource_type in self._SUCCESS_EVIDENCE_RESOURCE_TYPES:
			self._append_network_event(
				QANetworkEvent(
					request_id=request_id,
					method=self._request_methods.get(request_id, 'GET'),
					url=url,
					resource_type=resource_type,
					status=status_code,
				)
			)
		if status_code < 400:
			self._request_urls.pop(request_id, None)
			self._request_methods.pop(request_id, None)
			self._eligible_request_ids.discard(request_id)
			return
		self._append_network(f'HTTP {status_code} [{resource_type}] {url}')
		self._request_urls.pop(request_id, None)
		self._request_methods.pop(request_id, None)
		self._eligible_request_ids.discard(request_id)

	def _on_loading_failed(self, params: Any, _session_id: str | None) -> None:
		request_id = str(_get(params, 'requestId', ''))
		if self.navigation_scope is not None and request_id not in self._eligible_request_ids:
			return
		url = self._request_urls.get(request_id, 'unknown URL')
		error_text = _get(params, 'errorText', 'request failed')
		resource_type = _get(params, 'type', 'unknown')
		canceled = bool(_get(params, 'canceled', False))
		if canceled:
			self._request_urls.pop(request_id, None)
			self._request_methods.pop(request_id, None)
			self._eligible_request_ids.discard(request_id)
			return
		self._append_network_event(
			QANetworkEvent(
				request_id=request_id,
				method=self._request_methods.get(request_id, 'GET'),
				url=url,
				resource_type=str(resource_type),
				error=str(error_text),
			)
		)
		self._append_network(f'NETWORK_FAILED [{resource_type}] {url}: {error_text}')
		self._request_urls.pop(request_id, None)
		self._request_methods.pop(request_id, None)
		self._eligible_request_ids.discard(request_id)

	def _on_console(self, params: Any, _session_id: str | None) -> None:
		console_type = str(_get(params, 'type', '')).lower()
		if console_type not in {'error', 'assert'}:
			return
		parts: list[str] = []
		for argument in _get(params, 'args', []) or []:
			value = _get(argument, 'value')
			description = _get(argument, 'description')
			parts.append(str(value if value is not None else description or ''))
		self._append_console(f'console.{console_type}: {" ".join(parts)}')

	def _on_exception(self, params: Any, _session_id: str | None) -> None:
		details = _get(params, 'exceptionDetails', {})
		text = _get(details, 'text', 'Unhandled exception')
		exception = _get(details, 'exception', {})
		description = _get(exception, 'description')
		url = _get(details, 'url')
		self._append_console(f'UNHANDLED_EXCEPTION {url or "unknown URL"}: {description or text}')
