from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from bubus import EventBus

from browser_use.browser import BrowserProfile, BrowserSession
from browser_use.browser.events import BrowserStateRequestEvent
from browser_use.browser.watchdogs.security_watchdog import SecurityWatchdog
from browser_use.qa.compiler import QATaskCompiler, extract_task_urls
from browser_use.qa.navigation import NavigationScope


def test_extract_task_urls_preserves_order_and_requires_explicit_http_urls():
	task = 'Open https://foo.example.com/start, then verify https://api.example.com/status.'
	assert extract_task_urls(task) == ['https://foo.example.com/start', 'https://api.example.com/status']
	assert extract_task_urls('打开 https://foo.example.com/console，执行文章发布测试。') == ['https://foo.example.com/console']
	assert extract_task_urls('Open example.com and check it') == []


def test_extract_task_urls_handles_rich_text_markdown_links_without_caller_preprocessing():
	task = (
		'后台地址：<a href="[https://demo.halocms.site/console\\">]'
		'(https://demo.halocms.site/console\\">)'
		'<u>[https://demo.halocms.site/console](https://demo.halocms.site/console)</u></a>\n'
		'进入管理登录页<a href="[https://demo.halocms.site/console/dashboard\\">'
		'https://demo.halocms.site/console/dashboard]'
		'(https://demo.halocms.site/console/dashboard\\">https://demo.halocms.site/console/dashboard)</a>'
	)

	assert extract_task_urls(task) == [
		'https://demo.halocms.site/console',
		'https://demo.halocms.site/console/dashboard',
	]
	assert QATaskCompiler.resolve_scope(task).root_url == 'https://demo.halocms.site/console'


def test_navigation_scope_allows_registrable_domain_and_subdomains():
	scope = NavigationScope.from_root_url('https://foo.example.co.uk/app')
	assert scope.registrable_domain == 'example.co.uk'
	assert scope.allows('https://example.co.uk/')
	assert scope.allows('http://api.example.co.uk:8080/v1')
	assert not scope.allows('https://example.co.uk.evil.test/')
	assert not scope.allows('data:text/html,hello')
	assert not scope.allows('blob:https://foo.example.com/123')
	assert not scope.allows('file:///tmp/page.html')
	assert not scope.allows('chrome://settings')
	assert not scope.allows('https://user@example.com/app')
	assert not scope.allows('https://example.com.evil.test/app')
	assert scope.allows('http://api.example.co.uk:8080/health')


def test_navigation_scope_canonicalizes_idn_hosts():
	scope = NavigationScope.from_root_url('https://例子.公司.cn/app')
	assert scope.allows('https://xn--fsqu00a.xn--55qx5d.cn/other')
	assert not scope.allows('https://xn--fsqu00a.xn--55qx5d.cn.evil.test')
	assert not scope.allows('blob:https://foo.example.co.uk/id')


def test_navigation_scope_uses_private_suffixes_and_exact_local_hosts():
	private_scope = NavigationScope.from_root_url('https://tenant.github.io/app')
	assert private_scope.registrable_domain == 'tenant.github.io'
	assert private_scope.allows('https://sub.tenant.github.io/next')
	assert not private_scope.allows('https://other.github.io/')

	local_scope = NavigationScope.from_root_url('http://localhost:3000/app')
	assert local_scope.allows('https://localhost:9443/other')
	assert not local_scope.allows('http://sub.localhost:3000/')

	ipv4_scope = NavigationScope.from_root_url('http://127.0.0.1:8080/app')
	assert ipv4_scope.allows('https://127.0.0.1:9443/other')
	assert not ipv4_scope.allows('http://127.0.0.2:8080/')

	ipv6_scope = NavigationScope.from_root_url('http://[::1]:8080/app')
	assert ipv6_scope.allows('https://[::1]:9443/other')
	assert not ipv6_scope.allows('http://[::2]:8080/')


def test_compiler_rejects_missing_or_cross_domain_urls():
	try:
		QATaskCompiler.resolve_scope('Test the login form')
		assert False, 'missing URL must be rejected'
	except ValueError as exc:
		assert 'explicit HTTP(S)' in str(exc)

	try:
		QATaskCompiler.resolve_scope('Open https://example.com then https://evil.test')
		assert False, 'cross-domain task must be rejected'
	except ValueError as exc:
		assert 'outside the root domain' in str(exc)


def test_security_watchdog_intersects_qa_scope_with_existing_domain_rules():
	profile = BrowserProfile(
		qa_root_url='https://app.example.com/start',
		allowed_domains=['*.example.com'],
		prohibited_domains=['admin.example.com'],
		headless=True,
		user_data_dir=None,
	)
	session = BrowserSession(browser_profile=profile)
	watchdog = SecurityWatchdog(browser_session=session, event_bus=EventBus())

	assert watchdog._is_url_allowed('https://api.example.com/v1')
	assert not watchdog._is_url_allowed('https://admin.example.com/')
	assert not watchdog._is_url_allowed('https://example.com.evil.test/')
	assert not watchdog._is_url_allowed('file:///tmp/page.html')
	assert watchdog._is_url_allowed('about:blank')


def test_security_watchdog_does_not_change_normal_allowed_domain_precedence():
	profile = BrowserProfile(
		allowed_domains=['*.example.com'],
		prohibited_domains=['admin.example.com'],
		headless=True,
		user_data_dir=None,
	)
	session = BrowserSession(browser_profile=profile)
	watchdog = SecurityWatchdog(browser_session=session, event_bus=EventBus())
	assert watchdog._is_url_allowed('https://admin.example.com/')


@pytest.mark.asyncio
async def test_security_watchdog_blocks_click_or_spa_navigation_before_state_reaches_model(monkeypatch):
	page_navigate = AsyncMock()
	cdp_session = SimpleNamespace(
		session_id='session',
		cdp_client=SimpleNamespace(send=SimpleNamespace(Page=SimpleNamespace(navigate=page_navigate))),
	)
	session = BrowserSession(browser_profile=BrowserProfile(qa_root_url='https://example.com/app'))
	session.agent_focus_target_id = 'target'
	monkeypatch.setattr(BrowserSession, 'get_current_page_url', AsyncMock(return_value='https://evil.example.net/phishing'))
	monkeypatch.setattr(BrowserSession, 'get_or_create_cdp_session', AsyncMock(return_value=cdp_session))
	event_bus = EventBus()
	watchdog = SecurityWatchdog(browser_session=session, event_bus=event_bus)
	await watchdog.on_BrowserStateRequestEvent(BrowserStateRequestEvent())
	page_navigate.assert_awaited_once_with(params={'url': 'about:blank'}, session_id='session')
	await event_bus.stop(clear=True, timeout=1)
