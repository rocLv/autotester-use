import pytest
from pytest_httpserver import HTTPServer

from browser_use.agent.service import Agent
from browser_use.browser.events import NavigateToUrlEvent
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession
from browser_use.qa.compiler import QATaskCompiler
from browser_use.qa.views import ExpectationSource, WebUITestCase, WebUITestStep
from tests.ci.qa.test_qa_agent_integration import _qa_llm


@pytest.fixture(scope='session')
def http_server():
	"""Create and provide a test HTTP server for screenshot tests."""
	server = HTTPServer()
	server.start()

	# Route: Page with visible content for screenshot testing
	server.expect_request('/screenshot-page').respond_with_data(
		"""
		<!DOCTYPE html>
		<html>
		<head>
			<title>Screenshot Test Page</title>
			<style>
				body { font-family: Arial; padding: 20px; background: #f0f0f0; }
				h1 { color: #333; font-size: 32px; }
				.content { background: white; padding: 20px; border-radius: 8px; margin: 10px 0; }
			</style>
		</head>
		<body>
			<h1>Screenshot Test Page</h1>
			<div class="content">
				<p>This page is used to test screenshot capture with vision enabled.</p>
				<p>The agent should capture a screenshot when navigating to this page.</p>
			</div>
		</body>
		</html>
		""",
		content_type='text/html',
	)

	yield server
	server.stop()


@pytest.fixture(scope='session')
def base_url(http_server):
	"""Return the base URL for the test HTTP server."""
	return f'http://{http_server.host}:{http_server.port}'


@pytest.fixture(scope='function')
async def browser_session():
	session = BrowserSession(browser_profile=BrowserProfile(headless=True))
	await session.start()
	yield session
	await session.kill()


@pytest.mark.asyncio
async def test_basic_screenshots(browser_session: BrowserSession, httpserver):
	"""Navigate to a local page and ensure screenshot helpers return bytes."""

	html = """
    <html><body><h1 id='title'>Hello</h1><p>Screenshot demo.</p></body></html>
    """
	httpserver.expect_request('/demo').respond_with_data(html, content_type='text/html')
	url = httpserver.url_for('/demo')

	nav = browser_session.event_bus.dispatch(NavigateToUrlEvent(url=url, new_tab=False))
	await nav

	data = await browser_session.take_screenshot(full_page=False)
	assert data, 'Viewport screenshot returned no data'

	element = await browser_session.screenshot_element('h1')
	assert element, 'Element screenshot returned no data'


async def test_agent_screenshot_with_vision_enabled(browser_session, base_url):
	"""Test that agent captures screenshots when vision is enabled.

	This integration test verifies that:
	1. The QA runner with vision=True navigates to the declared root URL
	2. After prepare_context/update message manager, screenshot is captured
	3. Screenshot is included in the agent's history state
	"""
	root_url = f'{base_url}/screenshot-page'
	scope = QATaskCompiler.resolve_scope(f'Open {root_url}')
	test_case = WebUITestCase(
		root_url=root_url,
		registrable_domain=scope.registrable_domain,
		steps=[
			WebUITestStep(
				step_id='screenshot-heading',
				instruction='Inspect the screenshot test page heading',
				expected_result='The Screenshot Test Page heading is visible',
				expectation_source=ExpectationSource.UI_CONTRACT,
				source_evidence=['Visible h1 text: Screenshot Test Page'],
			)
		],
	)

	# Create agent with vision enabled
	agent = Agent(
		task=f'Open {root_url} and verify the Screenshot Test Page heading is visible.',
		llm=_qa_llm(),
		browser_session=browser_session,
		use_vision=True,  # Enable vision/screenshots
		qa_test_case=test_case,
	)

	# Run agent
	history = await agent.run(max_steps=2)

	# Verify agent completed successfully
	assert len(history) >= 1, 'Agent should have completed at least 1 step'
	final_result = history.final_result()
	assert final_result is not None, 'Agent should return a final result'

	# Verify screenshots were captured in the history
	screenshot_found = False
	for i, step in enumerate(history.history):
		# Check if browser state has screenshot path
		if step.state and hasattr(step.state, 'screenshot_path') and step.state.screenshot_path:
			screenshot_found = True
			print(f'\n✅ Step {i + 1}: Screenshot captured at {step.state.screenshot_path}')

			# Verify screenshot file exists (it should be saved to disk)
			import os

			assert os.path.exists(step.state.screenshot_path), f'Screenshot file should exist at {step.state.screenshot_path}'

			# Verify screenshot file has content
			screenshot_size = os.path.getsize(step.state.screenshot_path)
			assert screenshot_size > 0, f'Screenshot file should have content, got {screenshot_size} bytes'
			print(f'   Screenshot size: {screenshot_size} bytes')

	assert screenshot_found, 'At least one screenshot should be captured when vision is enabled'

	print('\n🎉 Integration test passed: Screenshots are captured correctly with vision enabled')
