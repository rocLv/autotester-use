import json

import pytest

import browser_use.agent.service as agent_service
from browser_use import Agent
from browser_use.qa.chrome_recorder import (
	ChromeRecorderFlow,
	ChromeRecorderPlaybackResult,
	ChromeRecorderPlaybackStatus,
	ChromeRecorderPlayer,
	ChromeRecorderStepResult,
	ChromeRecorderStepStatus,
	export_agent_history_to_chrome_recorder,
	load_chrome_recorder_flow,
	write_chrome_recorder_flow,
)
from tests.ci.conftest import create_mock_llm


class _FakeInputDomain:
	def __init__(self, calls):
		self.calls = calls

	async def dispatchMouseEvent(self, *, params, session_id):
		self.calls.append(('Input.dispatchMouseEvent', params, session_id))

	async def dispatchKeyEvent(self, *, params, session_id):
		self.calls.append(('Input.dispatchKeyEvent', params, session_id))


class _FakeRuntimeDomain:
	def __init__(self, calls, responses):
		self.calls = calls
		self.responses = list(responses)

	async def evaluate(self, *, params, session_id):
		self.calls.append(('Runtime.evaluate', params, session_id))
		if self.responses:
			return self.responses.pop(0)
		return {'result': {'value': {'ok': True, 'summary': 'ok', 'selector': 'css/#new'}}}


class _FakeEmulationDomain:
	def __init__(self, calls):
		self.calls = calls

	async def setDeviceMetricsOverride(self, *, params, session_id):
		self.calls.append(('Emulation.setDeviceMetricsOverride', params, session_id))


class _FakeSend:
	def __init__(self, calls, responses):
		self.Runtime = _FakeRuntimeDomain(calls, responses)
		self.Input = _FakeInputDomain(calls)
		self.Emulation = _FakeEmulationDomain(calls)


class _FakeCDPClient:
	def __init__(self, calls, responses):
		self.send = _FakeSend(calls, responses)


class _FakeCDPSession:
	def __init__(self, calls, responses):
		self.session_id = 'session-1'
		self.cdp_client = _FakeCDPClient(calls, responses)


class _FakeBrowserSession:
	def __init__(self, responses=None):
		self.calls = []
		self._cdp_session = _FakeCDPSession(self.calls, responses or [])

	async def start(self):
		self.calls.append(('start', {}, None))

	async def navigate_to(self, url, new_tab=False):
		self.calls.append(('navigate_to', {'url': url, 'new_tab': new_tab}, None))

	async def get_or_create_cdp_session(self):
		self.calls.append(('get_or_create_cdp_session', {}, None))
		return self._cdp_session


def test_chrome_recorder_flow_round_trip(tmp_path):
	path = tmp_path / 'recording.json'
	flow = ChromeRecorderFlow(
		title='Publish article',
		steps=[
			{'type': 'navigate', 'url': 'https://example.com/console'},
			{'type': 'click', 'selectors': [['aria/文章'], ['css/a[href="/posts"]']], 'assertedEvents': [{'type': 'navigation'}]},
		],
	)

	written = write_chrome_recorder_flow(flow, path)
	loaded = load_chrome_recorder_flow(written)

	assert loaded.title == 'Publish article'
	assert loaded.steps[1].selector_groups() == [['aria/文章'], ['css/a[href="/posts"]']]
	assert loaded.steps[1].model_extra['assertedEvents'] == [{'type': 'navigation'}]


@pytest.mark.asyncio
async def test_chrome_recorder_player_replays_common_steps():
	browser = _FakeBrowserSession(
		responses=[
			{'result': {'value': {'ok': True, 'summary': 'Clicked element', 'selector': 'aria/文章'}}},
			{'result': {'value': {'ok': True, 'summary': 'Changed element value', 'selector': 'css/#title'}}},
			{'result': {'value': {'ok': True, 'summary': 'Scrolled', 'x': 0, 'y': 700}}},
		]
	)
	player = ChromeRecorderPlayer(browser, default_timeout_ms=1)

	result = await player.replay(
		{
			'title': 'Replay article',
			'steps': [
				{'type': 'navigate', 'url': 'https://example.com/console'},
				{'type': 'click', 'selectors': [['aria/文章']]},
				{'type': 'change', 'selectors': [['css/#title']], 'value': 'hello'},
				{'type': 'scroll', 'y': 700},
				{'type': 'keyDown', 'key': 'Enter'},
				{'type': 'keyUp', 'key': 'Enter'},
				{'type': 'click', 'x': 20, 'y': 30},
				{'type': 'setViewport', 'width': 1280, 'height': 720},
			],
		}
	)

	assert result.status == ChromeRecorderPlaybackStatus.PASSED
	assert [step.status for step in result.steps] == [ChromeRecorderStepStatus.PASSED] * 8
	assert browser.calls[0] == ('navigate_to', {'url': 'https://example.com/console', 'new_tab': False}, None)
	assert any(call[0] == 'Runtime.evaluate' and '"action": "click"' in call[1]['expression'] for call in browser.calls)
	assert any(call[0] == 'Runtime.evaluate' and '"action": "change"' in call[1]['expression'] for call in browser.calls)
	assert any(call[0] == 'Input.dispatchKeyEvent' and call[1]['key'] == 'Enter' for call in browser.calls)
	assert any(call[0] == 'Input.dispatchMouseEvent' and call[1]['x'] == 20 and call[1]['y'] == 30 for call in browser.calls)
	assert any(call[0] == 'Emulation.setDeviceMetricsOverride' and call[1]['width'] == 1280 for call in browser.calls)


@pytest.mark.asyncio
async def test_chrome_recorder_player_reports_dom_failure_and_stops():
	browser = _FakeBrowserSession(
		responses=[
			{'result': {'value': {'ok': False, 'reason': 'element_not_found', 'tried': ['css/#missing']}}},
		]
	)
	player = ChromeRecorderPlayer(browser, default_timeout_ms=1)

	result = await player.replay(
		{
			'title': 'Missing element',
			'steps': [
				{'type': 'click', 'selectors': [['css/#missing']]},
				{'type': 'navigate', 'url': 'https://example.com/next'},
			],
		}
	)

	assert result.status == ChromeRecorderPlaybackStatus.FAILED
	assert len(result.steps) == 1
	assert result.steps[0].status == ChromeRecorderStepStatus.FAILED
	assert result.steps[0].selector == 'css/#missing'


def test_export_agent_history_to_chrome_recorder_uses_stable_selectors():
	history = [
		[
			{'navigate': {'url': 'https://example.com/console'}, 'interacted_element': None},
			{
				'click': {'index': 3},
				'interacted_element': {
					'attributes': {'id': 'posts-link', 'name': 'posts'},
					'x_path': '//*[@id="posts-link"]',
					'ax_name': '文章',
				},
			},
			{
				'input': {'index': 5, 'text': 'Title'},
				'interacted_element': {
					'attributes': {'id': 'article-title'},
					'x_path': '//*[@id="article-title"]',
				},
			},
			{'send_keys': {'keys': 'Enter'}, 'interacted_element': None},
		]
	]

	flow = export_agent_history_to_chrome_recorder(history, title='Exported')
	payload = json.loads(flow.model_dump_json(exclude_none=True))

	assert payload['title'] == 'Exported'
	assert [step['type'] for step in payload['steps']] == ['navigate', 'click', 'change', 'keyDown', 'keyUp']
	assert payload['steps'][1]['selectors'][0] == ['css/#posts-link']
	assert payload['steps'][1]['selectors'][1] == ['aria/文章']
	assert payload['steps'][2]['value'] == 'Title'


@pytest.mark.asyncio
async def test_agent_chrome_recording_helpers_delegate_to_core(tmp_path, monkeypatch):
	agent = Agent(task='Test https://example.com', llm=create_mock_llm())
	agent.browser_session = _FakeBrowserSession()
	monkeypatch.setattr(
		type(agent.history),
		'model_actions',
		lambda self: [
			{'navigate': {'url': 'https://example.com'}, 'interacted_element': None},
		],
	)

	flow = agent.export_chrome_recording(title='From agent')
	path = agent.save_chrome_recording(tmp_path / 'agent-recording.json', title='Saved agent')

	assert flow.title == 'From agent'
	assert flow.steps[0].url == 'https://example.com'
	assert load_chrome_recorder_flow(path).title == 'Saved agent'

	created = {}

	class _FakePlayer:
		def __init__(self, browser_session, **kwargs):
			created['browser_session'] = browser_session
			created['kwargs'] = kwargs

		async def replay(self, recording, *, max_steps=None):
			created['recording'] = recording
			created['max_steps'] = max_steps
			return ChromeRecorderPlaybackResult(
				title='Replay',
				status=ChromeRecorderPlaybackStatus.PASSED,
				steps=[
					ChromeRecorderStepResult(
						index=1,
						type='navigate',
						status=ChromeRecorderStepStatus.PASSED,
					)
				],
			)

	monkeypatch.setattr(agent_service, 'ChromeRecorderPlayer', _FakePlayer)
	result = await agent.replay_chrome_recording(
		{'title': 'Replay', 'steps': []},
		max_steps=1,
		default_timeout_ms=123,
		skip_unsupported=True,
	)

	assert result.status == ChromeRecorderPlaybackStatus.PASSED
	assert agent.browser_session.calls[0] == ('start', {}, None)
	assert created['browser_session'] is agent.browser_session
	assert created['kwargs']['default_timeout_ms'] == 123
	assert created['kwargs']['skip_unsupported'] is True
	assert created['max_steps'] == 1
