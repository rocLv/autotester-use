"""BrowserSession multi-tab creation, switching, closing, and state tests."""

import asyncio
import time

import pytest
from pytest_httpserver import HTTPServer

from browser_use.browser import BrowserSession
from browser_use.browser.events import CloseTabEvent, SwitchTabEvent
from browser_use.browser.profile import BrowserProfile


@pytest.fixture(scope='session')
def http_server():
	"""Create and provide a test HTTP server for tab tests."""

	server = HTTPServer()
	server.start()
	server.expect_request('/home').respond_with_data(
		'<html><head><title>Home Page</title></head><body><h1>Home Page</h1></body></html>',
		content_type='text/html',
	)
	server.expect_request('/page1').respond_with_data(
		'<html><head><title>Page 1</title></head><body><h1>Page 1</h1></body></html>',
		content_type='text/html',
	)
	server.expect_request('/page2').respond_with_data(
		'<html><head><title>Page 2</title></head><body><h1>Page 2</h1></body></html>',
		content_type='text/html',
	)
	server.expect_request('/page3').respond_with_data(
		'<html><head><title>Page 3</title></head><body><h1>Page 3</h1></body></html>',
		content_type='text/html',
	)
	server.expect_request('/background-tab-test').respond_with_data(
		"""
		<!doctype html><html><body>
		<a href="/page3" target="_blank" id="open-tab-link">Open New Tab</a>
		</body></html>
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
	"""Create a browser session for tab tests."""

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


class TestMultiTabOperations:
	"""Test multi-tab primitives independently of default Agent semantics."""

	@staticmethod
	async def _switch_to(browser_session: BrowserSession, target_id: str) -> None:
		event = browser_session.event_bus.dispatch(SwitchTabEvent(target_id=target_id))
		await event
		await event.event_result(raise_if_any=True, raise_if_none=False)

	@staticmethod
	async def _close(browser_session: BrowserSession, target_id: str) -> None:
		event = browser_session.event_bus.dispatch(CloseTabEvent(target_id=target_id))
		await event
		await event.event_result(raise_if_any=True, raise_if_none=False)
		for _ in range(20):
			if all(tab.target_id != target_id for tab in await browser_session.get_tabs()):
				return
			await asyncio.sleep(0.05)
		raise AssertionError(f'Closed tab {target_id} remained in the session cache')

	async def test_create_and_switch_three_tabs(self, browser_session: BrowserSession, base_url: str):
		"""Create three tabs, inspect state after each operation, and switch home."""

		started = time.time()
		try:
			await asyncio.wait_for(browser_session.navigate_to(f'{base_url}/home'), timeout=120)
			assert await browser_session.get_current_page_url() == f'{base_url}/home'
			await asyncio.wait_for(browser_session.navigate_to(f'{base_url}/page1', new_tab=True), timeout=120)
			assert await browser_session.get_current_page_url() == f'{base_url}/page1'
			await asyncio.wait_for(browser_session.navigate_to(f'{base_url}/page2', new_tab=True), timeout=120)
			assert await browser_session.get_current_page_url() == f'{base_url}/page2'

			tabs = await browser_session.get_tabs()
			assert len(tabs) >= 3
			home_tab = next(tab for tab in tabs if tab.url == f'{base_url}/home')
			await self._switch_to(browser_session, home_tab.target_id)
			assert await browser_session.get_current_page_url() == f'{base_url}/home'
			assert time.time() - started < 120
		except TimeoutError:
			pytest.fail('Timed out while creating or switching tabs')

	async def test_close_tab_with_vision(self, browser_session: BrowserSession, base_url: str):
		"""Capture a visible tab and close it without losing browser state."""

		await browser_session.navigate_to(f'{base_url}/home')
		await browser_session.navigate_to(f'{base_url}/page1', new_tab=True)
		page1 = next(tab for tab in await browser_session.get_tabs() if tab.url == f'{base_url}/page1')
		assert await browser_session.take_screenshot(full_page=False)
		await self._close(browser_session, page1.target_id)
		tabs = await browser_session.get_tabs()
		assert tabs
		assert all(tab.url != f'{base_url}/page1' for tab in tabs)

	async def test_background_tab_open_no_timeout(self, browser_session: BrowserSession, base_url: str):
		"""Opening another tab must not make cached browser state inaccessible."""

		await asyncio.wait_for(browser_session.navigate_to(f'{base_url}/home'), timeout=120)
		await asyncio.wait_for(browser_session.navigate_to(f'{base_url}/page1', new_tab=True), timeout=120)
		tabs = await asyncio.wait_for(browser_session.get_tabs(), timeout=120)
		assert len(tabs) >= 2
		assert await asyncio.wait_for(browser_session.get_current_page_url(), timeout=120)

	async def test_rapid_tab_operations_no_timeout(self, browser_session: BrowserSession, base_url: str):
		"""Browser state remains accessible while tabs are opened rapidly."""

		await asyncio.wait_for(browser_session.navigate_to(f'{base_url}/home'), timeout=120)
		for path in ('page1', 'page2', 'page3'):
			await asyncio.wait_for(browser_session.navigate_to(f'{base_url}/{path}', new_tab=True), timeout=120)
			assert await asyncio.wait_for(browser_session.get_current_page_url(), timeout=120) == f'{base_url}/{path}'
		assert len(await browser_session.get_tabs()) >= 4

	async def test_multiple_tab_switches_and_close(self, browser_session: BrowserSession, base_url: str):
		"""Switch among three tabs, close one, and retain coherent state."""

		await browser_session.navigate_to(f'{base_url}/home')
		await browser_session.navigate_to(f'{base_url}/page1', new_tab=True)
		await browser_session.navigate_to(f'{base_url}/page2', new_tab=True)
		page1 = next(tab for tab in await browser_session.get_tabs() if tab.url == f'{base_url}/page1')
		await self._switch_to(browser_session, page1.target_id)
		assert await browser_session.get_current_page_url() == f'{base_url}/page1'
		await self._close(browser_session, page1.target_id)
		tabs = await browser_session.get_tabs()
		assert len(tabs) >= 2
		assert all(tab.url != f'{base_url}/page1' for tab in tabs)
