from unittest.mock import AsyncMock, MagicMock

import pytest

from browser_use.qa.evidence import QAEvidenceMonitor
from browser_use.qa.navigation import NavigationScope


@pytest.mark.asyncio
async def test_evidence_monitor_collects_http_network_and_console_failures():
	cdp_session = MagicMock()
	cdp_session.session_id = 'session'
	cdp_session.cdp_client.send.Network.enable = AsyncMock()
	cdp_session.cdp_client.send.Runtime.enable = AsyncMock()
	browser_session = MagicMock()
	browser_session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
	browser_session.cdp_client.register = MagicMock()

	monitor = QAEvidenceMonitor(browser_session)
	await monitor.start()
	cursor = monitor.cursor()
	monitor._on_request({'requestId': '1', 'request': {'url': 'https://api.example.com/save'}}, None)
	monitor._on_response(
		{
			'requestId': '1',
			'type': 'Fetch',
			'response': {'status': 500, 'url': 'https://api.example.com/save'},
		},
		None,
	)
	monitor._on_loading_failed({'requestId': '2', 'type': 'XHR', 'errorText': 'net::ERR_TIMED_OUT'}, None)
	monitor._on_console({'type': 'error', 'args': [{'value': 'Save failed'}]}, None)
	monitor._on_exception(
		{'exceptionDetails': {'url': 'https://example.com/app.js', 'exception': {'description': 'TypeError: boom'}}}, None
	)

	diagnostics = monitor.since(cursor)
	assert diagnostics.network_errors == [
		'HTTP 500 [Fetch] https://api.example.com/save',
		'NETWORK_FAILED [XHR] unknown URL: net::ERR_TIMED_OUT',
	]
	assert diagnostics.console_errors == [
		'console.error: Save failed',
		'UNHANDLED_EXCEPTION https://example.com/app.js: TypeError: boom',
	]
	assert diagnostics.network_events[0].model_dump() == {
		'request_id': '1',
		'method': 'GET',
		'url': 'https://api.example.com/save',
		'resource_type': 'Fetch',
		'status': 500,
		'error': None,
	}


def test_evidence_monitor_ignores_successes_cancellations_and_console_info():
	monitor = QAEvidenceMonitor(MagicMock())
	cursor = monitor.cursor()
	monitor._on_response({'response': {'status': 204, 'url': 'https://example.com/save'}}, None)
	monitor._on_loading_failed({'requestId': '1', 'canceled': True}, None)
	monitor._on_console({'type': 'info', 'args': [{'value': 'loaded'}]}, None)
	assert monitor.since(cursor).network_errors == []
	assert monitor.since(cursor).console_errors == []


def test_evidence_monitor_allows_cross_domain_api_resources_but_ignores_other_tabs():
	monitor = QAEvidenceMonitor(MagicMock(), NavigationScope.from_root_url('https://app.example.com'))
	cursor = monitor.cursor()
	monitor._on_request(
		{
			'requestId': 'qa-api',
			'documentURL': 'https://app.example.com/settings',
			'type': 'Fetch',
			'request': {'url': 'https://external-api.test/save'},
		},
		None,
	)
	monitor._on_response(
		{'requestId': 'qa-api', 'type': 'Fetch', 'response': {'status': 503, 'url': 'https://external-api.test/save'}},
		None,
	)
	monitor._on_request(
		{
			'requestId': 'other-tab',
			'documentURL': 'https://unrelated.test',
			'type': 'Fetch',
			'request': {'url': 'https://external-api.test/noise'},
		},
		None,
	)
	monitor._on_response(
		{
			'requestId': 'other-tab',
			'type': 'Fetch',
			'response': {'status': 500, 'url': 'https://external-api.test/noise'},
		},
		None,
	)
	assert monitor.since(cursor).network_errors == ['HTTP 503 [Fetch] https://external-api.test/save']
