"""
Test navigation edge cases: broken pages, slow loading, non-existing pages.

Tests verify BrowserSession navigation independently from QA specification and model behavior.

Usage:
	uv run pytest tests/ci/browser/test_navigation.py -v -s
"""

import asyncio
import time

import pytest
from pytest_httpserver import HTTPServer
from werkzeug import Response

from browser_use.browser import BrowserSession
from browser_use.browser.profile import BrowserProfile


@pytest.fixture(scope='session')
def http_server():
	"""Create and provide a test HTTP server for navigation tests."""
	server = HTTPServer()
	server.start()

	# Route 1: Broken/malformed HTML page
	server.expect_request('/broken').respond_with_data(
		'<html><head><title>Broken Page</title></head><body><h1>Incomplete HTML',
		content_type='text/html',
	)

	# Route 2: Valid page for testing navigation after error recovery
	server.expect_request('/valid').respond_with_data(
		'<html><head><title>Valid Page</title></head><body><h1>Valid Page</h1><p>This page loaded successfully</p></body></html>',
		content_type='text/html',
	)

	# Route 3: Slow loading page - delays 10 seconds before responding
	def slow_handler(request):
		time.sleep(10)
		return Response(
			'<html><head><title>Slow Page</title></head><body><h1>Slow Loading Page</h1><p>This page took 10 seconds to load</p></body></html>',
			content_type='text/html',
		)

	server.expect_request('/slow').respond_with_handler(slow_handler)

	# Route 4: 404 page
	server.expect_request('/notfound').respond_with_data(
		'<html><head><title>404 Not Found</title></head><body><h1>404 - Page Not Found</h1></body></html>',
		status=404,
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
	"""Create a browser session for navigation tests."""
	session = BrowserSession(
		browser_profile=BrowserProfile(
			headless=True,
			user_data_dir=None,
			keep_alive=True,
		)
	)
	await session.start()
	yield session
	await session.kill()


class TestNavigationEdgeCases:
	"""Test navigation error handling and recovery."""

	async def test_broken_page_navigation(self, browser_session, base_url):
		"""Test that browser navigation tolerates broken/malformed HTML."""
		try:
			await asyncio.wait_for(browser_session.navigate_to(f'{base_url}/broken'), timeout=120)
			assert (await browser_session.get_current_page_url()).endswith('/broken')
		except TimeoutError:
			pytest.fail('Test timed out after 2 minutes - browser hung on broken page')

	async def test_slow_loading_page(self, browser_session, base_url):
		"""Test that agent can handle slow-loading pages without hanging."""
		start_time = time.time()
		try:
			await asyncio.wait_for(browser_session.navigate_to(f'{base_url}/slow'), timeout=120)
			elapsed = time.time() - start_time
			assert elapsed >= 10, f'Agent should have waited for slow page (10s delay), but only took {elapsed:.1f}s'
		except TimeoutError:
			pytest.fail('Test timed out after 2 minutes - browser hung on slow page')

	async def test_nonexisting_page_404(self, browser_session, base_url):
		"""Test that browser state remains observable on a 404 page."""
		try:
			await asyncio.wait_for(browser_session.navigate_to(f'{base_url}/notfound'), timeout=120)
			assert (await browser_session.get_current_page_url()).endswith('/notfound')
		except TimeoutError:
			pytest.fail('Test timed out after 2 minutes - browser hung on 404 page')

	async def test_nonexisting_domain(self, browser_session):
		"""Test that browser navigation to a refused connection returns without hanging."""

		# Use a localhost port that's not listening
		nonexisting_url = 'http://localhost:59999/page'

		try:
			with pytest.raises(RuntimeError, match='ERR_CONNECTION_REFUSED'):
				await asyncio.wait_for(browser_session.navigate_to(nonexisting_url), timeout=120)
		except TimeoutError:
			pytest.fail('Test timed out after 2 minutes - browser hung on non-existing domain')

	async def test_recovery_after_navigation_error(self, browser_session, base_url):
		"""Test that browser navigation can recover on a subsequent valid page."""
		try:
			await asyncio.wait_for(browser_session.navigate_to(f'{base_url}/broken'), timeout=120)
			await asyncio.wait_for(browser_session.navigate_to(f'{base_url}/valid'), timeout=120)
			final_url = await browser_session.get_current_page_url()
			assert final_url.endswith('/valid'), f'Final URL should be /valid, got {final_url}'
		except TimeoutError:
			pytest.fail('Test timed out after 2 minutes - browser could not recover from broken page')
