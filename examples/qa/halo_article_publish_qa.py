"""Run screenshot-derived QA cases on the public Halo demo.

The default case publishes one uniquely titled article. ``--compare-rerun``
uses a read-only article-list case and compares the first run with ``rerun()``.
The public demo may reset its data periodically.

Usage:

    uv run python examples/qa/halo_article_publish_qa.py
    uv run python examples/qa/halo_article_publish_qa.py --headed
    uv run python examples/qa/halo_article_publish_qa.py --compare-rerun
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, Field

from browser_use import Agent, BrowserProfile, BrowserSession, ChatBrowserUse, QARunStatus
from browser_use.qa import QAPhaseTiming

HALO_CONSOLE_URL = 'https://demo.halocms.site/console'
HALO_DOMAIN = 'demo.halocms.site'


class BenchmarkRunMetrics(BaseModel):
	"""Comparable measurements for one QA execution."""

	label: str
	reused_test_case: bool
	execution_mode: Literal['ai', 'replay']
	status: QARunStatus
	elapsed_seconds: float = Field(ge=0)
	llm_invocations: int = Field(ge=0)
	total_tokens: int = Field(ge=0)
	phase_timings: list[QAPhaseTiming] = Field(default_factory=list)


def build_test_data() -> tuple[str, str]:
	"""Create recognizable data that will not collide with an earlier demo run."""

	run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
	return (
		f'Browser Use QA 发布验证 {run_id}',
		f'这是 Browser Use Web UI QA 在 {run_id} 创建的自动化测试文章。',
	)


def build_task(article_title: str, article_content: str) -> str:
	"""Translate the four screenshot rows into explicit business QA steps."""

	return f"""打开 {HALO_CONSOLE_URL}，执行 Halo 文章发布测试。

前置条件：如果显示登录页，使用用户名 demo 和敏感数据 HALO_PASSWORD 登录，并进入管理后台。

严格按以下业务步骤执行；每一步只在预期结果可观察后才提交裁决：
1. 点击“文章”。预期结果：显示文章列表。
2. 点击右上角“新建”。预期结果：显示文章编辑页。
3. 输入文章标题“{article_title}”和正文内容“{article_content}”，点击编辑页的“发布”。预期结果：弹出文章设置弹窗。
4. 在文章设置弹窗内点击“发布”。预期结果：管理系统文章列表新增标题为“{article_title}”的已发布记录，并且 Halo 站点首页新增同名文章；必要时通过页面提供的站点入口查看首页以取得证据。
"""


def build_reuse_benchmark_task() -> str:
	"""Build a read-only real-site case that is safe to execute repeatedly."""

	return f"""打开 {HALO_CONSOLE_URL}，验证 Halo 后台文章列表。

前置条件：如果显示登录页，使用用户名 demo 和敏感数据 HALO_PASSWORD 登录，并进入管理后台。

严格执行以下业务步骤：
1. 点击侧边栏“文章”。预期结果：显示文章列表页面，并且页面上可见“新建”文章入口。
"""


def build_browser(*, headed: bool) -> BrowserSession:
	"""Create the domain-restricted browser shared by the demo modes."""

	return BrowserSession(
		browser_profile=BrowserProfile(
			headless=not headed,
			user_data_dir=None,
			# Keep the process and real authenticated session alive between the AI run
			# and its immediate deterministic replay. The finally blocks still kill it.
			keep_alive=True,
			allowed_domains=[HALO_DOMAIN],
		)
	)


def build_agent(*, task: str, browser: BrowserSession, use_vision: bool | Literal['auto'] = 'auto') -> Agent:
	"""Create the QA Agent with reusable credentials and specification caching."""

	return Agent(
		task=task,
		llm=ChatBrowserUse(),
		browser=browser,
		sensitive_data={
			HALO_DOMAIN: {
				'HALO_PASSWORD': 'P@ssw0rd123..',
			}
		},
		use_vision=use_vision,
		max_agent_retries_per_step=3,
		reuse_compiled_test_case=True,
		reuse_login_state=True,
	)


async def run_qa(*, headed: bool) -> int:
	"""Execute the real-site QA case and print the typed result."""

	article_title, article_content = build_test_data()
	browser = build_browser(headed=headed)
	agent = build_agent(task=build_task(article_title, article_content), browser=browser)
	try:
		history = await agent.run(max_steps=80)
		if history.qa_result is None:
			print('Agent returned no qa_result.')
			return 2
		print(history.qa_result.model_dump_json(indent=2))
		return 0 if history.qa_result.status == QARunStatus.PASSED else 1
	finally:
		await browser.kill()


async def run_reuse_benchmark(*, headed: bool) -> int:
	"""Execute the same safe real-site case twice and compare LLM/time efficiency."""

	browser = build_browser(headed=headed)
	# This benchmark only needs URL and semantic DOM evidence. Disabling screenshots
	# removes rendering/CDP screenshot variance from the two-run comparison.
	agent = build_agent(task=build_reuse_benchmark_task(), browser=browser, use_vision=False)
	metrics: list[BenchmarkRunMetrics] = []
	try:
		for run_number in (1, 2):
			usage_before = await agent.token_cost_service.get_usage_summary()
			started_at = perf_counter()
			history = await (agent.run(max_steps=30) if run_number == 1 else agent.rerun(max_steps=30, mode='replay'))
			elapsed_seconds = perf_counter() - started_at
			usage_after = await agent.token_cost_service.get_usage_summary()
			if history.qa_result is None:
				print(f'Run {run_number} returned no qa_result.')
				return 2
			metrics.append(
				BenchmarkRunMetrics(
					label='首次 AI 执行' if run_number == 1 else '第二次零 LLM 回放',
					reused_test_case=run_number == 2,
					execution_mode='ai' if run_number == 1 else 'replay',
					status=history.qa_result.status,
					elapsed_seconds=elapsed_seconds,
					llm_invocations=usage_after.entry_count - usage_before.entry_count,
					total_tokens=usage_after.total_tokens - usage_before.total_tokens,
					phase_timings=history.qa_result.phase_timings,
				)
			)
			if run_number == 1 and history.qa_result.test_case is not None:
				print('\n首次生成的强结构化用例：')
				print(history.qa_result.test_case.to_markdown_table())
			if run_number == 1 and history.qa_result.status != QARunStatus.PASSED:
				print('首次执行未通过，停止复用基准，避免比较无效结果。')
				return 1

		first, second = metrics
		time_reduction = 100 * (first.elapsed_seconds - second.elapsed_seconds) / first.elapsed_seconds
		invocation_reduction = first.llm_invocations - second.llm_invocations
		token_reduction = 100 * (first.total_tokens - second.total_tokens) / first.total_tokens if first.total_tokens else 0.0
		time_change_label = '降低' if time_reduction >= 0 else '增加'
		invocation_change_label = '减少' if invocation_reduction >= 0 else '增加'
		token_change_label = '降低' if token_reduction >= 0 else '增加'
		print('\n两次执行效率对比：')
		print('| 执行 | 模式 | 复用结构化用例 | QA 状态 | 耗时（秒） | LLM 调用数 | Token |')
		print('| --- | --- | --- | --- | ---: | ---: | ---: |')
		for item in metrics:
			print(
				f'| {item.label} | {item.execution_mode} | {"是" if item.reused_test_case else "否"} | '
				f'{item.status.value} | '
				f'{item.elapsed_seconds:.2f} | {item.llm_invocations} | {item.total_tokens} |'
			)
		print(
			f'\n复用结果：耗时{time_change_label} {abs(time_reduction):.1f}%，'
			f'{invocation_change_label} {abs(invocation_reduction)} 次 LLM 调用，'
			f'Token {token_change_label} {abs(token_reduction):.1f}%。'
		)
		if second.phase_timings:
			print('\n第二次零 LLM 回放耗时拆分：')
			print('| 阶段 | 耗时（秒） |')
			print('| --- | ---: |')
			for timing in second.phase_timings:
				print(f'| {timing.phase} | {timing.elapsed_seconds:.3f} |')
		return 0 if all(item.status == QARunStatus.PASSED for item in metrics) else 1
	finally:
		await browser.kill()


def parse_args() -> argparse.Namespace:
	"""Parse command-line options."""

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--headed', action='store_true', help='Show Chromium while the QA case executes')
	parser.add_argument(
		'--compare-rerun',
		action='store_true',
		help='Run a safe read-only case twice and compare specification/login reuse efficiency',
	)
	return parser.parse_args()


def main() -> int:
	"""Run the real Halo QA case."""

	args = parse_args()
	if args.compare_rerun:
		return asyncio.run(run_reuse_benchmark(headed=args.headed))
	return asyncio.run(run_qa(headed=args.headed))


if __name__ == '__main__':
	raise SystemExit(main())
