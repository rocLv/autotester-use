"""Watchdog for collecting same-site API calls into an OpenAPI document."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

from bubus import BaseEvent
from cdp_use.cdp.network.events import LoadingFinishedEvent, RequestWillBeSentEvent, ResponseReceivedEvent
from pydantic import Field, PrivateAttr

from browser_use.browser.api_collection import (
	ApiCollectionConfig,
	ApiSchemaCollector,
	is_json_content_type,
	registrable_domain,
	same_registrable_domain,
)
from browser_use.browser.events import BrowserConnectedEvent, BrowserStopEvent
from browser_use.browser.watchdog_base import BaseWatchdog


@dataclass
class _ApiRequestBuilder:
	request_id: str
	url: str | None = None
	method: str = 'GET'
	request_type: str | None = None
	document_url: str | None = None
	frame_id: str | None = None
	request_headers: dict[str, str] = dc_field(default_factory=dict)
	request_body: str | None = None
	response_headers: dict[str, str] = dc_field(default_factory=dict)
	response_content_type: str | None = None
	status_code: int | None = None
	session_id: str | None = None
	eligible: bool = False


class ApiCollectionWatchdog(BaseWatchdog):
	"""Collects API-like network traffic and exposes an OpenAPI 3.0 schema."""

	LISTENS_TO: ClassVar[list[type[BaseEvent]]] = [BrowserConnectedEvent, BrowserStopEvent]
	EMITS: ClassVar[list[type[BaseEvent]]] = []

	config: ApiCollectionConfig = Field(default_factory=ApiCollectionConfig)
	collector: ApiSchemaCollector = Field(default_factory=ApiSchemaCollector)
	_enabled: bool = PrivateAttr(default=False)
	_cdp_handlers_registered: bool = PrivateAttr(default=False)
	_requests: dict[str, _ApiRequestBuilder] = PrivateAttr(default_factory=dict)
	_frame_domains: dict[str, str] = PrivateAttr(default_factory=dict)
	_body_tasks: set[asyncio.Task] = PrivateAttr(default_factory=set)

	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.config = ApiCollectionConfig()
		self.collector = ApiSchemaCollector(config=self.config)

	async def start_collection(self, config: ApiCollectionConfig | None = None) -> None:
		"""Start collecting API traffic, enabling CDP Network listeners when connected."""
		self.config = config or ApiCollectionConfig()
		self.collector = ApiSchemaCollector(config=self.config)
		self._requests.clear()
		self._enabled = True
		if self.browser_session.is_cdp_connected:
			await self._enable_cdp_collection()

	def stop_collection(self) -> None:
		"""Stop accepting new network observations."""
		self._enabled = False

	def get_schema(self) -> dict[str, Any]:
		"""Return the current OpenAPI schema."""
		return self.collector.to_openapi()

	async def export_schema(self, path: str | Path | None = None) -> Path | None:
		"""Write the current OpenAPI schema to disk."""
		await self._drain_body_tasks()
		return self.collector.export(path)

	async def on_BrowserConnectedEvent(self, event: BrowserConnectedEvent) -> None:
		if self._enabled:
			await self._enable_cdp_collection()

	async def on_BrowserStopEvent(self, event: BrowserStopEvent) -> None:
		if not self._enabled:
			return
		await self._drain_body_tasks()
		if self.config.output_path:
			try:
				self.collector.export(self.config.output_path)
				self.logger.info(f'📡 API schema saved: {self.config.output_path}')
			except Exception as e:
				self.logger.warning(f'Failed to write API schema: {e}')

	async def _enable_cdp_collection(self) -> None:
		if self._cdp_handlers_registered:
			return
		try:
			cdp_session = await self.browser_session.get_or_create_cdp_session()
			await cdp_session.cdp_client.send.Network.enable(session_id=cdp_session.session_id)
			cdp = self.browser_session.cdp_client.register
			cdp.Network.requestWillBeSent(self._on_request_will_be_sent)
			cdp.Network.responseReceived(self._on_response_received)
			cdp.Network.loadingFinished(self._on_loading_finished)
			self._cdp_handlers_registered = True
			self.logger.info('📡 API collection started')
		except Exception as e:
			self.logger.warning(f'Failed to enable API collection: {e}')

	def _on_request_will_be_sent(self, params: RequestWillBeSentEvent, session_id: str | None) -> None:
		if not self._enabled:
			return
		try:
			req = _get_attr(params, 'request', {})
			url = _get_attr(req, 'url')
			request_id = _get_attr(params, 'requestId')
			if not request_id or not url:
				return

			request_type = str(_get_attr(params, 'type') or '')
			frame_id = _get_attr(params, 'frameId')
			document_url = _get_attr(params, 'documentURL')
			method = str(_get_attr(req, 'method') or 'GET').upper()
			headers = _normalize_headers(_get_attr(req, 'headers'))
			body = _get_attr(req, 'postData')

			if request_type == 'Document' and frame_id:
				domain = registrable_domain(urlparse(str(url)).hostname)
				if domain:
					self._frame_domains[str(frame_id)] = domain
				return

			builder = self._requests.setdefault(str(request_id), _ApiRequestBuilder(request_id=str(request_id)))
			builder.url = str(url)
			builder.method = method
			builder.request_type = request_type
			builder.document_url = str(document_url) if document_url else None
			builder.frame_id = str(frame_id) if frame_id else None
			builder.request_headers = headers
			builder.request_body = str(body) if body is not None else None
			builder.session_id = session_id
			builder.eligible = self._is_request_candidate(builder)
		except Exception as e:
			self.logger.debug(f'API requestWillBeSent handling error: {e}')

	def _on_response_received(self, params: ResponseReceivedEvent, session_id: str | None) -> None:
		if not self._enabled:
			return
		try:
			request_id = _get_attr(params, 'requestId')
			if not request_id or str(request_id) not in self._requests:
				return
			builder = self._requests[str(request_id)]
			response = _get_attr(params, 'response', {})
			builder.status_code = _as_int(_get_attr(response, 'status'))
			builder.response_headers = _normalize_headers(_get_attr(response, 'headers'))
			mime_type = _get_attr(response, 'mimeType')
			builder.response_content_type = (
				str(mime_type)
				or builder.response_headers.get('content-type')
				or builder.response_headers.get('Content-Type')
				or None
			)
			builder.session_id = session_id or builder.session_id
			builder.eligible = builder.eligible or is_json_content_type(builder.response_content_type)
		except Exception as e:
			self.logger.debug(f'API responseReceived handling error: {e}')

	def _on_loading_finished(self, params: LoadingFinishedEvent, session_id: str | None) -> None:
		if not self._enabled:
			return
		try:
			request_id = _get_attr(params, 'requestId')
			if not request_id or str(request_id) not in self._requests:
				return
			builder = self._requests[str(request_id)]
			if not self._should_collect(builder):
				return

			task = asyncio.create_task(self._record_response_body(builder, session_id or builder.session_id))
			self._body_tasks.add(task)
			task.add_done_callback(self._body_tasks.discard)
		except Exception as e:
			self.logger.debug(f'API loadingFinished handling error: {e}')

	async def _record_response_body(self, builder: _ApiRequestBuilder, session_id: str | None) -> None:
		response_body: str | bytes | None = None
		try:
			resp = await self.browser_session.cdp_client.send.Network.getResponseBody(
				params={'requestId': builder.request_id}, session_id=session_id
			)
			body = resp.get('body') if isinstance(resp, dict) else _get_attr(resp, 'body')
			base64_encoded = resp.get('base64Encoded') if isinstance(resp, dict) else _get_attr(resp, 'base64Encoded')
			if base64_encoded:
				import base64

				response_body = base64.b64decode(body or '')
			else:
				response_body = str(body) if body is not None else None
		except Exception:
			response_body = None

		if not builder.url:
			return
		self.collector.record(
			url=builder.url,
			method=builder.method,
			status_code=builder.status_code,
			request_headers=builder.request_headers,
			response_headers=builder.response_headers,
			request_body=builder.request_body,
			response_body=response_body,
			response_content_type=builder.response_content_type,
		)

	async def _drain_body_tasks(self) -> None:
		if not self._body_tasks:
			return
		await asyncio.gather(*list(self._body_tasks), return_exceptions=True)

	def _is_request_candidate(self, builder: _ApiRequestBuilder) -> bool:
		request_type = (builder.request_type or '').lower()
		if request_type in {'xhr', 'fetch'}:
			return True
		content_type = builder.request_headers.get('content-type') or builder.request_headers.get('Content-Type')
		return is_json_content_type(content_type)

	def _should_collect(self, builder: _ApiRequestBuilder) -> bool:
		if not builder.url or not builder.eligible:
			return False
		if (builder.request_type or '').lower() in {'document', 'stylesheet', 'image', 'media', 'font', 'script'}:
			return False
		if not self._looks_like_api_call(builder):
			return False
		if not self.config.same_site_only:
			return True
		page_url = builder.document_url
		if not page_url and builder.frame_id and builder.frame_id in self._frame_domains:
			return registrable_domain(urlparse(builder.url).hostname) == self._frame_domains[builder.frame_id]
		return same_registrable_domain(page_url, builder.url)

	def _looks_like_api_call(self, builder: _ApiRequestBuilder) -> bool:
		request_content_type = builder.request_headers.get('content-type')
		if is_json_content_type(request_content_type) or is_json_content_type(builder.response_content_type):
			return True
		path = urlparse(builder.url or '').path.lower()
		return path.startswith('/api/') or '/api/' in path or path.startswith('/graphql') or path.endswith('/graphql')


def _get_attr(value: Any, name: str, default: Any = None) -> Any:
	if isinstance(value, dict):
		return value.get(name, default)
	return getattr(value, name, default)


def _normalize_headers(headers_raw: Any) -> dict[str, str]:
	if headers_raw is None:
		return {}
	if isinstance(headers_raw, dict):
		return {str(k).lower(): str(v) for k, v in headers_raw.items()}
	if isinstance(headers_raw, list):
		return {
			str(item.get('name', '')).lower(): str(item.get('value') or '')
			for item in headers_raw
			if isinstance(item, dict) and item.get('name')
		}
	try:
		return {str(k).lower(): str(v) for k, v in dict(headers_raw).items()}
	except Exception:
		return {}


def _as_int(value: Any) -> int | None:
	try:
		return int(value)
	except (TypeError, ValueError):
		return None
