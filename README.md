<!-- mcp-name: com.roclv/autotester-use -->
<p align="center">
  <strong>English</strong> | <a href="README.zh-CN.md">简体中文</a>
</p>

# AutoTester Use

**A Web UI QA agent that turns natural-language test cases into typed specifications, executes one business step at a time, and distinguishes product defects from AI execution failures.**

AutoTester Use changes the default `browser_use.Agent` from a general task-completion agent into a Web UI QA runner. A run is successful only when the observable expectation of every business step is satisfied—not merely when the model says that it finished.

This QA behavior applies only to the default `browser_use.Agent`. `browser_use.beta.Agent` retains its existing behavior.

## Why use it?

- Compile a natural-language Task into a Pydantic `WebUITestCase` and print it as a Markdown table.
- Preserve explicit requirements with verified source quotes and character ranges.
- Judge every business step independently from objective browser evidence.
- Distinguish `SUT_FAILED`, `AGENT_FAILED`, `BLOCKED`, and `INCONCLUSIVE`.
- Require a runner-owned `ActionReceipt` and resolvable evidence IDs for reliable verdicts.
- Independently review every proposed product defect before reporting it.
- Reuse the compiled case and same-domain login state by default.
- Save versioned QA Bundles and replay reliable traces across processes.
- Run strict deterministic replay with zero LLM calls.
- Export JSON, JUnit XML, and self-contained HTML reports.
- Restrict interactive pages, tabs, and popups to the Task's registrable domain.

## Installation

Python 3.11 or newer is required.

```bash
uv add autotester-use
uv run autotester-use install
```

If you want Codex, Claude Code, Cursor, or another coding agent to use the packaged browser skill, run `autotester-use skill install` to register the skill.

For development from this repository:

```bash
uv sync
```

Configure an LLM. `ChatBrowserUse` is the recommended default for browser automation:

```bash
# .env
BROWSER_USE_API_KEY=your-key
```

AutoTester Use does not restrict model names. You can explicitly pass any model supported by the installed SDK.

## Complete runnable QA demo

This example uses the existing public Halo website. It compiles the Task with an LLM, verifies the article list, prints the typed case table, writes three report formats and a QA Bundle, then performs a strict zero-LLM replay when the first run produced a reliable pass.

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
Open {HALO_URL} and test the Halo article list.

Precondition: If the login page is displayed, sign in with username demo and
the sensitive value HALO_PASSWORD, then enter the administration console.

Business step:
1. Click "Articles" in the sidebar.
   Expected result: The article list is displayed and a "New" entry is visible.
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
            raise RuntimeError("Agent returned no qa_result")

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

The public demo may reset its data or credentials. Replace the URL, Task, and `sensitive_data` with your own test environment for stable CI.

Repository demos:

```bash
# Four-step article publishing case. This creates real data.
uv run python examples/qa/halo_article_publish_qa.py

# Safe read-only AI run followed by replay and timing comparison.
uv run python examples/qa/halo_article_publish_qa.py --compare-rerun

# Fixed-DOM performance ceiling with no Agent and no LLM.
uv run python examples/qa/halo_no_llm_baseline.py

# Fill the existing JJEBank registration form and solve its real slider; no SMS request or registration submission.
uv run python examples/qa/jjebank_register_slider_qa.py --headed

# Show the browser while running either Halo script.
uv run python examples/qa/halo_article_publish_qa.py --headed
```

## Task contract

Every Task passed to the default `browser_use.Agent` must:

- contain an explicit HTTP(S) start URL;
- describe business steps rather than only low-level browser calls;
- provide an observable expected result for each step whenever possible;
- describe login, roles, test data, and dependencies as preconditions; and
- stay within one registrable domain for all interactive pages.

The model compiles the Task into a strongly typed document before formal execution. Explicit wording is verified against the original Task or `ground_truth`; the model cannot invent a requirement quote. If an expected result is missing, AutoTester Use can perform read-only discovery in a disposable CDP BrowserContext and mark the inferred source as `ui_contract` or `heuristic`.

```python
result = history.qa_result
assert result is not None

if result.test_case:
    print(result.test_case.to_markdown_table())
```

The generated table includes the root URL, registrable domain, typed preconditions, operation kind, side-effect level, idempotency key, expected result, expectation source, and verified requirement references.

Expectation sources have different authority:

| Source | Meaning | Can directly support `SUT_FAILED`? |
| --- | --- | --- |
| `explicit` | Directly stated in the Task or `ground_truth` | Yes |
| `ui_contract` | Supported by labels, ARIA, HTML validation, or another observable UI contract | Yes |
| `heuristic` | Model inference without a reliable contract | No; use `INCONCLUSIVE` |

## Reliable verdicts and failure attribution

Every result uses `schema_version=2` and is available through `history.qa_result`.

| Status | Origin | Meaning |
| --- | --- | --- |
| `PASSED` | `none` | The intended actions completed and every reliable expectation was met. |
| `SUT_FAILED` | `sut` | The intended action completed, a reliable expectation was not met, and independent review agreed. |
| `AGENT_FAILED` | `agent` | The AI selected the wrong target, omitted an action, exhausted safe retries, or a required model failed. |
| `BLOCKED` | `environment` | Credentials, role, CAPTCHA, browser/CDP, network, test data, or URL policy blocked execution. |
| `INCONCLUSIVE` | `unknown` | Evidence was weak, missing, conflicting, or the expectation was only heuristic. |
| `INVALID_SPEC` | — | The URL or typed test structure was invalid; formal browser execution did not start. |

`history.is_successful()` deliberately preserves the difference between product and execution failures:

```python
history.is_successful()  # PASSED -> True; SUT_FAILED -> False; otherwise -> None
```

A reliable pass or product failure requires:

1. an `ActionReceipt` built by the runner from actual tool results;
2. at least one resolvable `evidence_id`;
3. a completed intended action;
4. a reliable `explicit` or `ui_contract` expectation for product failures; and
5. an independent second review for every proposed `SUT_FAILED`.

The Judge cannot change the receipt's action-completion status. Reviewer failure or disagreement never remains a product-defect verdict.

Inspect the attribution directly:

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

Execution stops at the first reliable non-pass outcome. Remaining business steps are recorded as `NOT_RUN` to avoid cascading defect reports.

## Preconditions, login reuse, and side effects

Preconditions are typed as:

- `VERIFY`: inspect a role, page state, dependency, or test-data condition without changing it;
- `ENSURE`: perform an explicitly allowed setup action such as login, separately from business steps.

Required preconditions that cannot be satisfied produce `BLOCKED`, not `SUT_FAILED`. Login state is retained by default only for the current Agent and navigation scope. It is revalidated before reuse and is not copied to another registrable domain.

Side-effecting operations receive a run-scoped idempotency key. When a submit, publish, delete, payment, or similar action may already have taken effect, AutoTester Use gathers more evidence instead of blindly retrying. If the effect remains uncertain, the result is `INCONCLUSIVE`.

Cleanup is optional and recorded separately. Cleanup failure becomes a warning and never overwrites the business verdict.

## Rerun and replay modes

```python
# Reuse the compiled test case and login state, but execute and judge with AI again.
ai_rerun = await agent.rerun(mode="ai")

# Replay deterministic actions; use AI only when a target or observation drifts.
adaptive_replay = await agent.rerun(mode="replay")

# Strict replay: guaranteed zero LLM calls.
strict_replay = await agent.rerun(
    mode="replay",
    allow_llm_fallback=False,
)
```

| Mode | LLM behavior | Recommended use |
| --- | --- | --- |
| `mode="ai"` | Executor, Judge, and required review calls remain | New or highly dynamic UI |
| `mode="replay"` | Zero LLM while the trace matches; audited fallback for only the drifted step | Fast adaptive regression |
| strict replay | Always zero LLM; drift cannot fall back | Stable UI, deterministic CI, cost ceilings |

Replay fallback is allowed only when the target drifted, the recorded action cannot be replayed, or required evidence is no longer observable. If the action completed and a reliable assertion is observably false, the normal product-verdict path runs immediately; AI fallback cannot hide a product defect.

Every result records `requested_mode`, `effective_mode`, fallback warnings, `llm_call_count`, and phase timings.

## Versioned QA Bundles and cross-process replay

A QA Bundle is a local checksummed directory containing:

```text
manifest.json
test_case.json
run_result.json
actions/<step_id>.json
artifacts/<evidence_id>.*
```

Save and load a bundle:

```python
from browser_use import Agent, ChatBrowserUse, QABundle

bundle = agent.save_qa_bundle("./artifacts/halo.qa-bundle")

# In a later process. Configure persistent authentication separately when needed.
fresh_agent = Agent(task=TASK, llm=ChatBrowserUse())
strict_history = await fresh_agent.rerun(
    mode="replay",
    bundle=QABundle.load(bundle.path),
    allow_llm_fallback=False,
)
```

Loading rejects checksum damage, incompatible schemas, an unverified legacy result, Task or `ground_truth` mismatch, root URL mismatch, and registrable-domain mismatch before formal business-step execution. AI-repaired traces are appended as immutable revisions only after a reliable pass.

Bundles do not store reusable secrets. For authenticated cross-process replay, configure a persistent `user_data_dir`, an existing browser profile, or a hosted browser profile.

## Reports

```python
agent.save_qa_report("./reports/run.json")
agent.save_qa_report("./reports/junit.xml")
agent.save_qa_report("./reports/run.html")
```

JUnit mapping:

- `SUT_FAILED` → test failure;
- `AGENT_FAILED`, `BLOCKED`, `INCONCLUSIVE`, `INVALID_SPEC` → execution error;
- `NOT_RUN` → skipped.

The HTML report includes typed steps, before/after evidence, action receipts, independent reviews, cleanup outcomes, phase timings, and redacted diagnostics. Sensitive values, cookies, authorization headers, and response bodies are not written as raw evidence.

## Pass an already typed test case

Use `qa_test_case` to skip natural-language compilation when the caller owns a validated specification. Explicit expectations still need an exact source reference.

```python
from browser_use import Agent, ChatBrowserUse, WebUITestCase, WebUITestStep
from browser_use.qa import (
    ExpectationSource,
    RequirementReference,
    RequirementSource,
    SideEffectLevel,
    StepOperationKind,
)

TASK = """Open https://example.com and run one test step.
Step: Inspect the page heading.
Expected result: The heading Example Domain is visible."""
REQUIREMENT = "Expected result: The heading Example Domain is visible."
start = TASK.index(REQUIREMENT)

test_case = WebUITestCase(
    root_url="https://example.com",
    registrable_domain="example.com",
    steps=[
        WebUITestStep(
            step_id="heading_visible",
            instruction="Inspect the page heading.",
            expected_result="The heading Example Domain is visible.",
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

The Task start URL and registrable domain must match the supplied case.

## URL safety

- The Task must contain one explicit HTTP(S) start URL.
- The start URL's registrable domain and its subdomains form the maximum navigation scope.
- Caller-provided `allowed_domains` and `prohibited_domains` can only narrow that scope.
- Requested navigation, final redirects, newly created tabs, popups, and SPA URL changes are checked.
- `localhost` and IP addresses use exact-host matching.
- Credentials in URLs and interactive `data:`, `blob:`, `file:`, or `chrome:` pages are rejected.
- Cross-domain XHR, API, CDN, image, and other subresources remain available and can contribute evidence.

`use_judge=False` and non-empty `initial_actions` are rejected by the default QA Agent. `max_agent_retries_per_step` accepts `0..3` and defaults to `3`.

## Performance guidance

- Use `headless=True` for CI and batch execution; use headed mode for debugging and visual observation.
- Keep `reuse_compiled_test_case=True` to skip unchanged Task compilation.
- Keep `reuse_login_state=True` to avoid repeated login within the same Agent and domain.
- Use strict replay for stable regression paths that need zero LLM cost.
- Use `use_vision=False` only when URL and semantic DOM evidence are sufficient.
- Inspect `result.phase_timings` instead of relying on total wall-clock time alone.
- For production browser performance, `Browser(use_cloud=True)` provisions a hosted browser optimized for automation and requires `BROWSER_USE_API_KEY`.

The fixed-DOM script in `examples/qa/halo_no_llm_baseline.py` is an optimization ceiling, not a general QA agent: it is faster because it gives up natural-language compilation, semantic adaptation, independent model judgment, and generic failure attribution.

## Development

```bash
uv sync
uv run pytest tests/ci/qa
uv run pre-commit run --all-files
```

## Documentation and license

- [Chinese README](README.zh-CN.md)
- [Complete Halo QA demo](examples/qa/halo_article_publish_qa.py)
- [Zero-LLM fixed-DOM baseline](examples/qa/halo_no_llm_baseline.py)
- [JJEBank form and slider QA demo](examples/qa/jjebank_register_slider_qa.py)
- [Upstream browser automation documentation](https://docs.browser-use.com)
- [Hosted browser service](https://cloud.browser-use.com)

AutoTester Use is licensed under the MIT License. Linked hosted services have their own terms and privacy policies.
