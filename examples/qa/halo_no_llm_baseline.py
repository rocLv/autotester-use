"""Benchmark the Halo article-list QA flow without Agent or any LLM.

This is an optimization-ceiling baseline for ``halo_article_publish_qa.py``.
It uses fixed DOM contracts for the public Halo demo, keeps one browser alive,
and runs the same read-only check twice so the second run reuses authentication.

Usage:

    uv run python examples/qa/halo_no_llm_baseline.py
    uv run python examples/qa/halo_no_llm_baseline.py --headed
"""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, Field

from browser_use import BrowserProfile, BrowserSession
from browser_use.actor.element import Element
from browser_use.actor.page import Page

HALO_CONSOLE_URL = 'https://demo.halocms.site/console'
HALO_DOMAIN = 'demo.halocms.site'


class PhaseTiming(BaseModel):
	"""Wall-clock timing for one deterministic browser phase."""

	phase: str = Field(min_length=1)
	elapsed_seconds: float = Field(ge=0)


class DeterministicRunMetrics(BaseModel):
	"""Result and timing of one no-LLM Halo run."""

	label: str
	status: Literal['PASSED', 'FAILED']
	elapsed_seconds: float = Field(ge=0)
	login_performed: bool
	llm_invocations: Literal[0] = 0
	total_tokens: Literal[0] = 0
	phases: list[PhaseTiming] = Field(default_factory=list)
	error: str | None = None


@contextmanager
def measure_phase(timings: list[PhaseTiming], phase: str) -> Iterator[None]:
	"""Append one measured phase without hiding exceptions."""

	started_at = perf_counter()
	try:
		yield
	finally:
		timings.append(PhaseTiming(phase=phase, elapsed_seconds=perf_counter() - started_at))


async def wait_until(
	page: Page,
	predicate: str,
	*,
	timeout: float = 15.0,
	poll_interval: float = 0.1,
	description: str,
) -> None:
	"""Poll a fixed JavaScript predicate until the UI contract is observable."""

	deadline = perf_counter() + timeout
	while perf_counter() < deadline:
		if (await page.evaluate(predicate)).lower() == 'true':
			return
		await asyncio.sleep(poll_interval)
	raise TimeoutError(f'Timed out waiting for {description}')


async def first_visible(elements: list[Element]) -> Element | None:
	"""Return the first attached element with a non-empty layout box."""

	for element in elements:
		box = await element.get_bounding_box()
		if box is not None and box['width'] > 0 and box['height'] > 0:
			return element
	return None


async def login_if_needed(page: Page, *, username: str, password: str) -> bool:
	"""Login only when the fixed Halo login form is present."""

	login_visible = await page.evaluate("() => Boolean(document.querySelector('input[type=password], input[name=password]'))")
	if login_visible.lower() != 'true':
		return False

	username_inputs = await page.get_elements_by_css_selector('input[name=username], input[autocomplete=username]')
	password_inputs = await page.get_elements_by_css_selector('input[name=password], input[type=password]')
	buttons = await page.get_elements_by_css_selector('button')
	username_input = await first_visible(username_inputs)
	password_input = await first_visible(password_inputs)
	login_button = None
	for button in buttons:
		box = await button.get_bounding_box()
		if (
			box is not None
			and box['width'] > 0
			and box['height'] > 0
			and (await button.evaluate('() => this.textContent.trim()')) == '登录'
		):
			login_button = button
			break
	if username_input is None or password_input is None or login_button is None:
		raise RuntimeError('Halo login controls were not found')
	await username_input.fill(username)
	await password_input.fill(password)
	await login_button.click()
	await wait_until(
		page,
		"() => location.pathname.startsWith('/console') && !document.querySelector('input[type=password]')",
		timeout=20.0,
		description='authenticated Halo console',
	)
	return True


async def click_articles(browser: BrowserSession, page: Page) -> None:
	"""Click the fixed Halo sidebar contract for the Articles page."""

	await wait_until(
		page,
		"""() => [...document.querySelectorAll('a, button, [role=menuitem], [role=button], div')]
			.some((element) => element.textContent.trim() === '文章' && element.getClientRects().length > 0)""",
		description='visible Articles navigation entry',
	)
	state = await browser.get_browser_state_summary(include_screenshot=False)
	candidates = [
		node for node in state.dom_state.selector_map.values() if node.get_meaningful_text_for_llm() == '文章' and node.is_visible
	]
	if not candidates:
		raise RuntimeError('Articles navigation entry was not found')
	target = max(
		candidates,
		key=lambda node: (
			bool(node.snapshot_node and node.snapshot_node.is_clickable),
			bool(node.ax_node and node.ax_node.role in {'link', 'menuitem', 'button'}),
			node.node_name.lower() in {'a', 'button', 'div'},
		),
	)
	await (await page.get_element(target.backend_node_id)).click()


async def verify_article_list(page: Page) -> None:
	"""Verify the same explicit expected result used by the Agent benchmark."""

	await wait_until(
		page,
		"""() => location.pathname.startsWith('/console/posts') &&
			[...document.querySelectorAll('button, a, [role=button]')]
				.some((element) => element.textContent.trim() === '新建' && element.getClientRects().length > 0)""",
		description='article list URL and visible New entry',
	)


async def navigate_to_root(browser: BrowserSession, *, attempts: int = 3) -> None:
	"""Retry only the idempotent root GET when Chromium reports a transient network change."""

	for attempt in range(1, attempts + 1):
		try:
			await browser.navigate_to(HALO_CONSOLE_URL, new_tab=False)
			return
		except RuntimeError as exc:
			if attempt == attempts or not any(
				marker in str(exc) for marker in ('ERR_NETWORK_CHANGED', 'ERR_CONNECTION_RESET', 'ERR_TIMED_OUT')
			):
				raise
			await asyncio.sleep(0.5 * attempt)


async def run_once(browser: BrowserSession, *, run_number: int, username: str, password: str) -> DeterministicRunMetrics:
	"""Run the no-LLM business step once and return typed metrics."""

	timings: list[PhaseTiming] = []
	started_at = perf_counter()
	login_performed = False
	try:
		if run_number == 1:
			with measure_phase(timings, 'browser_start'):
				await browser.start()
		with measure_phase(timings, 'root_navigation'):
			await navigate_to_root(browser)
			page = await browser.must_get_current_page()
			await wait_until(
				page,
				"() => document.readyState === 'complete'",
				description='Halo root document',
			)
		with measure_phase(timings, 'login_if_needed'):
			login_performed = await login_if_needed(page, username=username, password=password)
		with measure_phase(timings, 'click_articles'):
			await click_articles(browser, page)
		with measure_phase(timings, 'verify_expected_result'):
			await verify_article_list(page)
		return DeterministicRunMetrics(
			label='首次无 LLM 执行' if run_number == 1 else '第二次无 LLM 执行（复用登录）',
			status='PASSED',
			elapsed_seconds=perf_counter() - started_at,
			login_performed=login_performed,
			phases=timings,
		)
	except Exception as exc:
		return DeterministicRunMetrics(
			label='首次无 LLM 执行' if run_number == 1 else '第二次无 LLM 执行（复用登录）',
			status='FAILED',
			elapsed_seconds=perf_counter() - started_at,
			login_performed=login_performed,
			phases=timings,
			error=f'{type(exc).__name__}: {exc}',
		)


def print_metrics(metrics: list[DeterministicRunMetrics]) -> None:
	"""Print comparable tables for the no-LLM baseline."""

	print('| 执行 | QA 状态 | 登录 | 耗时（秒） | LLM 调用数 | Token |')
	print('| --- | --- | --- | ---: | ---: | ---: |')
	for item in metrics:
		print(
			f'| {item.label} | {item.status} | {"是" if item.login_performed else "复用"} | '
			f'{item.elapsed_seconds:.2f} | {item.llm_invocations} | {item.total_tokens} |'
		)
		if item.error:
			print(f'错误：{item.error}')
	print('\n阶段耗时：')
	print('| 执行 | 阶段 | 耗时（秒） |')
	print('| --- | --- | ---: |')
	for item in metrics:
		for timing in item.phases:
			print(f'| {item.label} | {timing.phase} | {timing.elapsed_seconds:.3f} |')


async def run_benchmark(*, headed: bool, password_provider: Callable[[], str]) -> int:
	"""Run two deterministic passes in one browser with retained login state."""

	browser = BrowserSession(
		browser_profile=BrowserProfile(
			headless=not headed,
			user_data_dir=None,
			keep_alive=True,
			allowed_domains=[HALO_DOMAIN],
			minimum_wait_page_load_time=0.1,
			wait_for_network_idle_page_load_time=0.1,
			wait_between_actions=0.1,
		)
	)
	metrics: list[DeterministicRunMetrics] = []
	try:
		for run_number in (1, 2):
			metrics.append(
				await run_once(
					browser,
					run_number=run_number,
					username='demo',
					password=password_provider(),
				)
			)
			if metrics[-1].status != 'PASSED':
				break
		print_metrics(metrics)
		return 0 if len(metrics) == 2 and all(item.status == 'PASSED' for item in metrics) else 1
	finally:
		await browser.kill()


def parse_args() -> argparse.Namespace:
	"""Parse command-line options."""

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--headed', action='store_true', help='Show Chromium during the deterministic benchmark')
	return parser.parse_args()


def main() -> int:
	"""Run the no-LLM benchmark without requiring an API key."""

	args = parse_args()
	return asyncio.run(
		run_benchmark(
			headed=args.headed,
			password_provider=lambda: os.environ.get('HALO_PASSWORD', 'P@ssw0rd123..'),
		)
	)


if __name__ == '__main__':
	raise SystemExit(main())
