<!-- mcp-name: com.roclv/autotester-use -->
<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

# AutoTester Use

**一个面向 Web UI 的 QA Agent：把自然语言测试用例转换为强类型规格，逐业务步骤执行和裁决，并区分产品缺陷与 AI 执行失败。**

AutoTester Use 将默认 `browser_use.Agent` 从通用的“任务完成 Agent”改造为 Web UI QA Runner。只有当每个业务步骤的可观察预期都得到满足时，用例才算通过；执行模型自述“已完成”不能作为测试结论。

QA 行为只应用于默认 `browser_use.Agent`，`browser_use.beta.Agent` 保持原有行为。

## 核心能力

- 使用大模型把自然语言 Task 编译为 Pydantic `WebUITestCase`，并输出 Markdown 表格。
- 保存显式需求的原文引用和字符位置，避免模型伪造需求来源。
- 每个业务步骤由独立 Judge 根据客观浏览器证据裁决。
- 明确区分 `SUT_FAILED`、`AGENT_FAILED`、`BLOCKED` 和 `INCONCLUSIVE`。
- 可靠结论必须具备 Runner 生成的 `ActionReceipt` 和有效 `evidence_id`。
- 所有拟判定的产品缺陷都进行第二次独立复核。
- 默认复用强结构化用例和同主域登录状态。
- 支持版本化 QA Bundle 和跨进程轨迹回放。
- 支持严格零 LLM 的确定性 replay。
- 生成 JSON、JUnit XML 和自包含 HTML 报告。
- 将可交互页面、标签页和弹窗限制在 Task 的注册主域范围内。

## 安装

需要 Python 3.11 或更高版本。

```bash
uv add autotester-use
uv run autotester-use install
```

如需让 Codex、Claude Code、Cursor 等编码 Agent 使用随包提供的浏览器 Skill，请执行：

```bash
autotester-use skill install
```

在当前源码仓库开发：

```bash
uv sync
```

配置模型。浏览器自动化默认推荐 `ChatBrowserUse`：

```bash
# .env
BROWSER_USE_API_KEY=your-key
```

AutoTester Use 不限制模型名称，调用方可以显式传入当前 SDK 支持的任意模型。

## 完整可运行 QA Demo

下面的示例使用现成的 Halo 公开网站。它会调用大模型编译 Task、验证文章列表、打印强结构化用例表格、生成三种报告和一个 QA Bundle；如果首次运行可靠通过，还会执行一次严格零 LLM replay。

```python
import asyncio

from dotenv import load_dotenv

from browser_use import (
    Agent,
    BrowserProfile,
    BrowserSession,
    ChatBrowserUse,
    QARunStatus,
)

load_dotenv()

HALO_URL = "https://demo.halocms.site/console"
HALO_DOMAIN = "demo.halocms.site"

TASK = f"""
打开 {HALO_URL}，测试 Halo 后台文章列表。

前置条件：如果显示登录页，使用用户名 demo 和敏感数据 HALO_PASSWORD 登录，
然后进入管理后台。

业务步骤：
1. 点击侧边栏“文章”。
   预期结果：显示文章列表，并且页面上可见“新建”入口。
"""


async def main() -> None:
    browser = BrowserSession(
        browser_profile=BrowserProfile(
            headless=True,
            keep_alive=True,
            user_data_dir=None,
            allowed_domains=[HALO_DOMAIN],
        )
    )
    agent = Agent(
        task=TASK,
        llm=ChatBrowserUse(),
        browser=browser,
        sensitive_data={
            HALO_DOMAIN: {
                "HALO_PASSWORD": "P@ssw0rd123..",
            }
        },
        max_agent_retries_per_step=3,
        reuse_compiled_test_case=True,
        reuse_login_state=True,
    )

    try:
        history = await agent.run(max_steps=30)
        result = history.qa_result
        if result is None:
            raise RuntimeError("Agent 没有返回 qa_result")

        if result.test_case:
            print(result.test_case.to_markdown_table())
        print(result.model_dump_json(indent=2))

        agent.save_qa_report("./artifacts/halo-run.json")
        agent.save_qa_report("./artifacts/halo-junit.xml")
        agent.save_qa_report("./artifacts/halo-report.html")
        agent.save_qa_bundle("./artifacts/halo.qa-bundle")

        if result.status == QARunStatus.PASSED and result.has_reliable_verdict:
            replay_history = await agent.rerun(
                max_steps=30,
                mode="replay",
                allow_llm_fallback=False,
            )
            replay_result = replay_history.qa_result
            assert replay_result is not None
            assert replay_result.llm_call_count == 0
            print(replay_result.model_dump_json(indent=2))
    finally:
        await browser.kill()


if __name__ == "__main__":
    asyncio.run(main())
```

公开 Demo 可能定期重置数据或账号。用于稳定 CI 时，请替换为自己的测试环境 URL、Task 和 `sensitive_data`。

仓库 Demo：

```bash
# 四步骤发布文章用例，会真实创建数据。
uv run python examples/qa/halo_article_publish_qa.py

# 安全的只读 AI 首次执行、replay 和耗时对比。
uv run python examples/qa/halo_article_publish_qa.py --compare-rerun

# 不依赖 Agent 和 LLM 的固定 DOM 性能上限。
uv run python examples/qa/halo_no_llm_baseline.py

# 在现成的 JJEBank 注册页填写表单并完成真实滑动验证；不会获取短信或提交注册。
uv run python examples/qa/jjebank_register_slider_qa.py --headed

# 显示浏览器窗口。
uv run python examples/qa/halo_article_publish_qa.py --headed
```

## Task 编写契约

传给默认 `browser_use.Agent` 的 Task 必须：

- 包含明确的 HTTP(S) 起始 URL；
- 描述业务步骤，而不是只描述底层浏览器调用；
- 尽可能为每个步骤提供可观察的预期结果；
- 将登录、角色、测试数据和依赖写为前置条件；
- 所有可交互页面都处于同一个注册主域范围内。

正式执行前，大模型会把 Task 编译为强类型文档。显式需求会与 Task 或 `ground_truth` 原文校验，模型不能凭空生成需求引用。缺少预期时，AutoTester Use 可以在一次性 CDP BrowserContext 中执行只读探索，并把推断来源标记为 `ui_contract` 或 `heuristic`。

```python
result = history.qa_result
assert result is not None

if result.test_case:
    print(result.test_case.to_markdown_table())
```

表格包含起始 URL、注册主域、强类型前置条件、操作类型、副作用级别、幂等标识、预期结果、预期来源和经过验证的需求引用。

不同预期来源具有不同可信度：

| 来源 | 含义 | 能否直接支持 `SUT_FAILED` |
| --- | --- | --- |
| `explicit` | Task 或 `ground_truth` 明确给出的需求 | 可以 |
| `ui_contract` | 标签、ARIA、HTML 校验或其他可观察页面契约 | 可以 |
| `heuristic` | 没有可靠契约支持的模型推断 | 不可以，只能 `INCONCLUSIVE` |

## 可信裁决与失败归因

所有结果使用 `schema_version=2`，通过 `history.qa_result` 读取。

| 状态 | 失败来源 | 含义 |
| --- | --- | --- |
| `PASSED` | `none` | 预期动作已经完成，并且所有可靠预期都得到满足。 |
| `SUT_FAILED` | `sut` | 预期动作已经完成，可靠预期不满足，而且二次独立复核一致。 |
| `AGENT_FAILED` | `agent` | AI 选错目标、遗漏操作、安全重试耗尽，或必要模型调用失败。 |
| `BLOCKED` | `environment` | 凭证、角色、验证码、浏览器/CDP、网络、测试数据或 URL 策略阻塞执行。 |
| `INCONCLUSIVE` | `unknown` | 证据较弱、缺失、冲突，或者预期只有启发式依据。 |
| `INVALID_SPEC` | 无 | URL 或强类型测试结构无效，正式浏览器执行不会启动。 |

`history.is_successful()` 不会混淆产品失败和执行失败：

```python
history.is_successful()  # PASSED -> True；SUT_FAILED -> False；其他 -> None
```

可靠的通过或产品失败必须同时具备：

1. Runner 根据真实工具结果生成的 `ActionReceipt`；
2. 至少一个可以解析的 `evidence_id`；
3. 目标业务动作已经确认完成；
4. 产品失败具有 `explicit` 或 `ui_contract` 可靠预期；
5. 所有拟判定的 `SUT_FAILED` 都完成第二次独立复核。

Judge 不能修改回执中的动作完成状态。复核模型不可用或两次结论不一致时，不会保留产品缺陷结论。

读取逐步骤归因：

```python
for step_result in result.step_results:
    print(step_result.step.step_id, step_result.status)
    print(step_result.attempt_receipts)
    if step_result.judgement:
        print(step_result.judgement.failure_origin)
        print(step_result.judgement.failure_code)
        print(step_result.judgement.evidence_ids)
    if step_result.review:
        print(step_result.review.agreed, step_result.review.reason)
```

出现第一个可靠的非通过结果后立即停止，剩余业务步骤标记为 `NOT_RUN`，避免产生级联缺陷。

## 前置条件、登录复用与副作用

前置条件分为：

- `VERIFY`：只检查角色、页面状态、依赖或测试数据，不改变它们；
- `ENSURE`：执行明确允许的准备动作，例如登录，并与业务步骤分开记录。

必需前置条件无法满足时返回 `BLOCKED`，不能产生 `SUT_FAILED`。登录状态默认只在当前 Agent 和导航范围内保留，复用前会重新验证，也不会复制到另一个注册主域。

写操作会获得当前 `run_id` 范围内的幂等标识。当提交、发布、删除、支付等操作可能已经生效时，AutoTester Use 会优先补充取证，而不是盲目重试；仍然无法确认时返回 `INCONCLUSIVE`。

Cleanup 是可选的，结果单独记录。Cleanup 失败只形成警告，不覆盖原业务结论。

## 再次执行与 Replay 模式

```python
# 复用结构化用例和登录状态，但重新调用执行模型与 Judge。
ai_rerun = await agent.rerun(mode="ai")

# 回放确定性动作；只有目标或证据漂移时才让 AI 修复对应步骤。
adaptive_replay = await agent.rerun(mode="replay")

# 严格 replay：保证零 LLM 调用。
strict_replay = await agent.rerun(
    mode="replay",
    allow_llm_fallback=False,
)
```

| 模式 | LLM 行为 | 推荐场景 |
| --- | --- | --- |
| `mode="ai"` | 保留执行模型、Judge 和必要的复核调用 | 新页面或高度动态 UI |
| `mode="replay"` | 轨迹匹配时零 LLM；只对漂移步骤进行可审计 AI 回退 | 快速且能适应变化的回归 |
| 严格 replay | 始终零 LLM，发生漂移也不回退 | 稳定 UI、确定性 CI、成本上限 |

只有目标漂移、记录动作无法回放或必要证据不再可观察时，replay 才允许回退 AI。如果动作已经完成，而可靠断言明确不满足，会立即进入正常产品裁决流程；AI 回退不能掩盖产品缺陷。

每次结果都会记录 `requested_mode`、`effective_mode`、回退警告、`llm_call_count` 和阶段耗时。

## 版本化 QA Bundle 与跨进程 Replay

QA Bundle 是一个本地校验和目录：

```text
manifest.json
test_case.json
run_result.json
actions/<step_id>.json
artifacts/<evidence_id>.*
```

保存和加载 Bundle：

```python
from browser_use import Agent, ChatBrowserUse, QABundle

bundle = agent.save_qa_bundle("./artifacts/halo.qa-bundle")

# 在后续 Python 进程中运行；需要登录时应单独配置持久化认证状态。
fresh_agent = Agent(task=TASK, llm=ChatBrowserUse())
strict_history = await fresh_agent.rerun(
    mode="replay",
    bundle=QABundle.load(bundle.path),
    allow_llm_fallback=False,
)
```

Bundle 损坏、schema 不兼容、旧版结论未重新验证、Task 或 `ground_truth` 不匹配、根 URL 不匹配、注册主域不匹配，都会在正式业务步骤开始前失败。AI 修复后的轨迹只有可靠通过时才追加为不可变修订，不覆盖旧轨迹。

Bundle 不保存可复用的敏感凭证。认证场景的跨进程 replay 需要配置持久化 `user_data_dir`、已有浏览器 Profile 或托管浏览器 Profile。

## 报告

```python
agent.save_qa_report("./reports/run.json")
agent.save_qa_report("./reports/junit.xml")
agent.save_qa_report("./reports/run.html")
```

JUnit 映射：

- `SUT_FAILED` → 测试 failure；
- `AGENT_FAILED`、`BLOCKED`、`INCONCLUSIVE`、`INVALID_SPEC` → 执行 error；
- `NOT_RUN` → skipped。

HTML 报告包含强类型步骤、前后证据、动作回执、二次复核、cleanup 结果、阶段耗时和脱敏诊断。敏感值、Cookie、认证头和响应体不会作为原始证据写入文件。

## 直接传入强类型用例

如果调用方已经拥有验证过的测试规格，可以使用 `qa_test_case` 跳过自然语言编译。显式预期仍然必须提供精确的原文引用。

```python
from browser_use import Agent, ChatBrowserUse, WebUITestCase, WebUITestStep
from browser_use.qa import (
    ExpectationSource,
    RequirementReference,
    RequirementSource,
    SideEffectLevel,
    StepOperationKind,
)

TASK = """打开 https://example.com，执行一个测试步骤。
步骤：检查页面标题。
预期结果：页面上可见 Example Domain 标题。"""
REQUIREMENT = "预期结果：页面上可见 Example Domain 标题。"
start = TASK.index(REQUIREMENT)

test_case = WebUITestCase(
    root_url="https://example.com",
    registrable_domain="example.com",
    steps=[
        WebUITestStep(
            step_id="heading_visible",
            instruction="检查页面标题。",
            expected_result="页面上可见 Example Domain 标题。",
            expectation_source=ExpectationSource.EXPLICIT,
            operation_kind=StepOperationKind.OBSERVE,
            side_effect_level=SideEffectLevel.NONE,
            requirement_references=[
                RequirementReference(
                    source=RequirementSource.TASK,
                    quote=REQUIREMENT,
                    start=start,
                    end=start + len(REQUIREMENT),
                )
            ],
        )
    ],
)

agent = Agent(task=TASK, qa_test_case=test_case, llm=ChatBrowserUse())
```

Task 的起始 URL 和注册主域必须与传入用例一致。

## URL 安全

- Task 必须包含一个明确的 HTTP(S) 起始 URL。
- 起始 URL 的注册主域及其子域构成最大导航范围。
- 调用方传入的 `allowed_domains` 和 `prohibited_domains` 只能进一步收窄范围。
- 请求导航、最终重定向、新标签页、弹窗和 SPA URL 变化都会接受检查。
- `localhost` 和 IP 地址只允许精确主机匹配。
- 拒绝 URL 中的凭证，以及可交互的 `data:`、`blob:`、`file:`、`chrome:` 页面。
- XHR、API、CDN、图片等跨域子资源可以正常加载，并可参与证据收集。

默认 QA Agent 会拒绝 `use_judge=False` 和非空 `initial_actions`。`max_agent_retries_per_step` 取值为 `0..3`，默认 `3`。

## 性能建议

- CI 和批量执行使用 `headless=True`；调试和观察交互过程时使用 headed 模式。
- 保持 `reuse_compiled_test_case=True`，避免重复编译未变化的 Task。
- 保持 `reuse_login_state=True`，避免同一 Agent 和主域内重复登录。
- 稳定回归路径使用严格 replay，实现零 LLM 成本。
- 只有 URL 和语义 DOM 证据已经足够时才设置 `use_vision=False`。
- 使用 `result.phase_timings` 分析各阶段耗时，不要只看总耗时。
- 生产环境可以使用 `Browser(use_cloud=True)` 自动创建针对浏览器自动化优化的托管浏览器，需要 `BROWSER_USE_API_KEY`。

`examples/qa/halo_no_llm_baseline.py` 是固定 DOM 性能上限，不是通用 QA Agent。它更快，是因为放弃了自然语言编译、语义适应、独立模型裁决和通用失败归因。

## 开发验证

```bash
uv sync
uv run pytest tests/ci/qa
uv run pre-commit run --all-files
```

## 文档与许可

- [English README](README.md)
- [完整 Halo QA Demo](examples/qa/halo_article_publish_qa.py)
- [零 LLM 固定 DOM 基线](examples/qa/halo_no_llm_baseline.py)
- [JJEBank 表单与滑动验证 QA Demo](examples/qa/jjebank_register_slider_qa.py)
- [上游浏览器自动化文档](https://docs.browser-use.com)
- [托管浏览器服务](https://cloud.browser-use.com)

AutoTester Use 使用 MIT License。链接的托管服务有各自的服务条款和隐私政策。
