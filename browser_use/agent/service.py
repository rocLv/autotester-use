import asyncio
import base64
import gc
import hashlib
import inspect
import json
import logging
import re
import tempfile
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast
from urllib.parse import urlparse

if TYPE_CHECKING:
	from browser_use.skills.views import Skill

from dotenv import load_dotenv

from browser_use.agent.cloud_events import (
	CreateAgentOutputFileEvent,
	CreateAgentSessionEvent,
	CreateAgentStepEvent,
	CreateAgentTaskEvent,
	UpdateAgentTaskEvent,
)
from browser_use.agent.message_manager.utils import save_conversation
from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelOutputTruncatedError, ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import BaseMessage, ContentPartImageParam, ContentPartTextParam, UserMessage
from browser_use.tokens.service import TokenCost

load_dotenv()

from bubus import EventBus
from pydantic import BaseModel, ValidationError
from uuid_extensions import uuid7str

from browser_use import Browser, BrowserProfile, BrowserSession
from browser_use.agent.judge import construct_judge_messages

# Lazy import for gif to avoid heavy agent.views import at startup
# from browser_use.agent.gif import create_history_gif
from browser_use.agent.message_manager.service import (
	MessageManager,
)
from browser_use.agent.prompts import SystemPrompt
from browser_use.agent.views import (
	ActionResult,
	AgentError,
	AgentHistory,
	AgentHistoryList,
	AgentOutput,
	AgentSettings,
	AgentState,
	AgentStepInfo,
	AgentStructuredOutput,
	BrowserStateHistory,
	DetectedVariable,
	JudgementResult,
	MessageCompactionSettings,
	PlanItem,
	StepMetadata,
)
from browser_use.browser.events import BrowserStartEvent, ScrollEvent, SwitchTabEvent, _get_timeout
from browser_use.browser.session import DEFAULT_BROWSER_PROFILE
from browser_use.browser.views import BrowserError, BrowserStateSummary
from browser_use.config import CONFIG
from browser_use.dom.views import DOMInteractedElement, MatchLevel
from browser_use.filesystem.file_system import FileSystem
from browser_use.observability import observe, observe_debug
from browser_use.qa.bundle import QABundle, qa_content_hash
from browser_use.qa.chrome_recorder import (
	ChromeRecorderFlow,
	ChromeRecorderPlaybackResult,
	ChromeRecorderPlayer,
	export_agent_history_to_chrome_recorder,
	write_agent_history_chrome_recorder_flow,
)
from browser_use.qa.compiler import QATaskCompiler, WebUITestCaseDraft
from browser_use.qa.evidence import QAEvidenceCursor, QAEvidenceMonitor
from browser_use.qa.judge import judge_test_step, replay_assertion_matches
from browser_use.qa.navigation import NavigationScope
from browser_use.qa.views import (
	ActionCompletionStatus,
	ActionExpectationProof,
	ActionReceipt,
	ActionTargetProof,
	BrowserEvidenceSnapshot,
	EvidenceArtifact,
	EvidenceKind,
	EvidenceQuality,
	ExpectationSource,
	ExpectationStatus,
	FailureCode,
	FailureOrigin,
	FinishTestStepAction,
	PreconditionMode,
	PreconditionStatus,
	QACleanupResult,
	QAPhaseTiming,
	QAPlanSnapshot,
	QAPlanStep,
	QAPreconditionResult,
	QARunResult,
	QARunStatus,
	QAStepResult,
	QAStepStatus,
	ReviewRecord,
	SideEffectLevel,
	StepEvidence,
	StepJudgement,
	StepOperationKind,
	WebUITestCase,
	WebUITestStep,
)
from browser_use.telemetry.service import ProductTelemetry
from browser_use.telemetry.views import AgentTelemetryEvent
from browser_use.tools.registry.views import ActionModel
from browser_use.tools.service import Tools
from browser_use.utils import (
	URL_PATTERN,
	_log_pretty_path,
	check_latest_browser_use_version,
	collect_sensitive_data_values,
	get_browser_use_version,
	is_placeholder_url,
	redact_sensitive_string,
	sanitize_url_candidate,
	time_execution_async,
	time_execution_sync,
)

logger = logging.getLogger(__name__)


def _message_content_for_llm_log(content: object) -> str:
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		return '\n'.join(part.text if isinstance(part, ContentPartTextParam) else str(part) for part in content)
	return str(content)


def _messages_for_llm_log(messages: list[BaseMessage]) -> str:
	return '\n\n'.join(f'{message.role}:\n{_message_content_for_llm_log(message.content)}' for message in messages)


def _model_output_for_llm_log(value: object) -> str:
	return str(value)


def log_response(response: AgentOutput, registry=None, logger=None) -> None:
	"""Utility function to log the model's response."""

	# Use module logger if no logger provided
	if logger is None:
		logger = logging.getLogger(__name__)

	# Only log thinking if it's present
	if response.current_state.thinking:
		logger.debug(f'💡 Thinking:\n{response.current_state.thinking}')

	# Only log evaluation if it's not empty
	eval_goal = response.current_state.evaluation_previous_goal
	if eval_goal:
		if 'success' in eval_goal.lower():
			emoji = '👍'
			# Green color for success
			logger.info(f'  \033[32m{emoji} Eval: {eval_goal}\033[0m')
		elif 'failure' in eval_goal.lower():
			emoji = '⚠️'
			# Red color for failure
			logger.info(f'  \033[31m{emoji} Eval: {eval_goal}\033[0m')
		else:
			emoji = '❔'
			# No color for unknown/neutral
			logger.info(f'  {emoji} Eval: {eval_goal}')

	# Always log memory if present
	if response.current_state.memory:
		logger.info(f'  🧠 Memory: {response.current_state.memory}')

	# Only log next goal if it's not empty
	next_goal = response.current_state.next_goal
	if next_goal:
		# Blue color for next goal
		logger.info(f'  \033[34m🎯 Next goal: {next_goal}\033[0m')


Context = TypeVar('Context')


AgentHookFunc = Callable[['Agent'], Awaitable[None]]


class QAExplorationStateError(RuntimeError):
	"""Raised when read-only discovery state cannot be safely restored."""

	def __init__(self, message: str, *, inconclusive: bool = False):
		super().__init__(message)
		self.inconclusive = inconclusive


class QAIsolatedContextUnavailable(RuntimeError):
	"""Raised when CDP cannot provide a usable isolated exploration context."""


class Agent(Generic[Context, AgentStructuredOutput]):
	@time_execution_sync('--init')
	def __init__(
		self,
		task: str,
		llm: BaseChatModel | None = None,
		# Optional parameters
		browser_profile: BrowserProfile | None = None,
		browser_session: BrowserSession | None = None,
		browser: Browser | None = None,  # Alias for browser_session
		tools: Tools[Context] | None = None,
		controller: Tools[Context] | None = None,  # Alias for tools
		# Skills integration
		skill_ids: list[str | Literal['*']] | None = None,
		skills: list[str | Literal['*']] | None = None,  # Alias for skill_ids
		skill_service: Any | None = None,
		# Initial agent run parameters
		sensitive_data: dict[str, str | dict[str, str]] | None = None,
		initial_actions: list[dict[str, dict[str, Any]]] | None = None,
		# Cloud Callbacks
		register_new_step_callback: (
			Callable[['BrowserStateSummary', 'AgentOutput', int], None]  # Sync callback
			| Callable[['BrowserStateSummary', 'AgentOutput', int], Awaitable[None]]  # Async callback
			| None
		) = None,
		register_done_callback: (
			Callable[['AgentHistoryList'], Awaitable[None]]  # Async Callback
			| Callable[['AgentHistoryList'], None]  # Sync Callback
			| None
		) = None,
		register_qa_plan_callback: (
			Callable[[QAPlanSnapshot], Awaitable[Any]]  # Async Callback
			| Callable[[QAPlanSnapshot], Any]  # Sync Callback
			| None
		) = None,
		register_external_agent_status_raise_error_callback: Callable[[], Awaitable[bool]] | None = None,
		register_should_stop_callback: Callable[[], Awaitable[bool]] | None = None,
		# Agent settings
		output_model_schema: type[AgentStructuredOutput] | None = None,
		extraction_schema: dict | None = None,
		use_vision: bool | Literal['auto'] = True,
		save_conversation_path: str | Path | None = None,
		save_conversation_path_encoding: str | None = 'utf-8',
		max_failures: int = 5,
		override_system_message: str | None = None,
		extend_system_message: str | None = None,
		generate_gif: bool | str = False,
		available_file_paths: list[str] | None = None,
		include_attributes: list[str] | None = None,
		max_actions_per_step: int = 5,
		use_thinking: bool = True,
		flash_mode: bool = False,
		demo_mode: bool | None = None,
		max_history_items: int | None = None,
		page_extraction_llm: BaseChatModel | None = None,
		fallback_llm: BaseChatModel | None = None,
		use_judge: bool = True,
		ground_truth: str | None = None,
		judge_llm: BaseChatModel | None = None,
		review_llm: BaseChatModel | None = None,
		injected_agent_state: AgentState | None = None,
		source: str | None = None,
		file_system_path: str | None = None,
		task_id: str | None = None,
		calculate_cost: bool = False,
		pricing_url: str | None = None,
		display_files_in_done_text: bool = True,
		include_tool_call_examples: bool = False,
		vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
		llm_timeout: int | None = None,
		step_timeout: int = 180,
		directly_open_url: bool = True,
		include_recent_events: bool = False,
		sample_images: list[ContentPartTextParam | ContentPartImageParam] | None = None,
		final_response_after_failure: bool = True,
		enable_planning: bool = True,
		planning_replan_on_stall: int = 3,
		planning_exploration_limit: int = 5,
		loop_detection_window: int = 20,
		loop_detection_enabled: bool = True,
		llm_screenshot_size: tuple[int, int] | None = None,
		message_compaction: MessageCompactionSettings | bool | None = True,
		max_clickable_elements_length: int = 40000,
		max_agent_retries_per_step: int = 3,
		qa_test_case: WebUITestCase | None = None,
		reuse_compiled_test_case: bool = True,
		reuse_login_state: bool = True,
		qa_step_limit: int | None = None,
		_url_shortening_limit: int = 25,
		enable_signal_handler: bool = True,
		**kwargs,
	):
		if not use_judge:
			raise ValueError('browser_use.Agent is a Web UI QA agent and requires use_judge=True')
		if initial_actions:
			raise ValueError('initial_actions are not supported in QA mode; describe setup as a test step with an expectation')
		if not 0 <= max_agent_retries_per_step <= 3:
			raise ValueError('max_agent_retries_per_step must be between 0 and 3')
		if qa_step_limit is not None and qa_step_limit < 1:
			raise ValueError('qa_step_limit must be at least 1 when provided')

		self._qa_original_task = task
		self._qa_scope: NavigationScope | None = None
		self._qa_scope_error: str | None = None
		try:
			self._qa_scope = QATaskCompiler.resolve_scope(task)
		except ValueError as exc:
			self._qa_scope_error = str(exc)
		self._completion_action_name = 'finish_test_step'
		if qa_test_case is not None and self._qa_scope is not None:
			if qa_test_case.root_url != self._qa_scope.root_url:
				raise ValueError('qa_test_case.root_url must match the explicit Task start URL')
			if qa_test_case.registrable_domain != self._qa_scope.registrable_domain:
				raise ValueError('qa_test_case.registrable_domain must match the Task navigation scope')
		self._qa_test_case: WebUITestCase | None = qa_test_case
		self._qa_compiled_task: str | None = task if qa_test_case is not None else None
		self._qa_test_case_draft: WebUITestCaseDraft | None = None
		self._qa_login_storage_state: dict[str, Any] | None = None
		self._qa_current_step_index = 0
		self._qa_step_retry_count = 0
		self._qa_current_attempt_receipts: list[ActionReceipt] = []
		self._qa_step_results: list[QAStepResult] = []
		self._qa_step_history_start = 0
		self._qa_before_snapshot: BrowserEvidenceSnapshot | None = None
		self._qa_evidence_monitor: QAEvidenceMonitor | None = None
		self._qa_evidence_cursor: QAEvidenceCursor | None = None
		self._qa_replay_history: dict[str, list[AgentHistory]] = {}
		self._qa_phase_timings: dict[str, float] = {}
		self._qa_phase = 'specification'
		self._qa_done_callback_called = False
		self._qa_llm_call_count = 0
		self._qa_requested_mode: Literal['ai', 'replay'] = 'ai'
		self._qa_effective_mode: Literal['ai', 'replay', 'ai_fallback'] = 'ai'
		self._qa_precondition_results = []
		self._qa_warnings: list[str] = []
		self._qa_running_preconditions = False
		self._qa_precondition_step: WebUITestStep | None = None
		self._qa_run_id = uuid7str()
		self._qa_cleanup_results: list[QACleanupResult] = []
		self._qa_running_cleanup = False
		self._qa_cleanup_index = 0
		self._qa_pending_business_outcome: dict[str, Any] | None = None
		self._qa_replay_allow_llm_fallback = True
		self._qa_replay_fallback_active = False
		self._qa_replay_fallback_completed = False
		# Validate llm_screenshot_size
		if llm_screenshot_size is not None:
			if not isinstance(llm_screenshot_size, tuple) or len(llm_screenshot_size) != 2:
				raise ValueError('llm_screenshot_size must be a tuple of (width, height)')
			width, height = llm_screenshot_size
			if not isinstance(width, int) or not isinstance(height, int):
				raise ValueError('llm_screenshot_size dimensions must be integers')
			if width < 100 or height < 100:
				raise ValueError('llm_screenshot_size dimensions must be at least 100 pixels')
			self.logger.info(f'🖼️  LLM screenshot resizing enabled: {width}x{height}')
		if llm is None:
			default_llm_name = CONFIG.DEFAULT_LLM
			if default_llm_name:
				from browser_use.llm.models import get_llm_by_name

				llm = get_llm_by_name(default_llm_name)
			else:
				# No default LLM specified, use the original default
				from browser_use import ChatBrowserUse

				llm = ChatBrowserUse()

		# set flashmode = True if llm is ChatBrowserUse
		if llm.provider == 'browser-use':
			flash_mode = True

		# Flash mode strips plan fields from the output schema, so planning is structurally impossible
		if flash_mode:
			enable_planning = False

		# Auto-configure llm_screenshot_size for Claude Sonnet, including gateway ids like
		# 'anthropic/claude-sonnet-4-6' (rsplit drops the provider prefix before matching).
		if llm_screenshot_size is None:
			model_name = getattr(llm, 'model', '')
			if isinstance(model_name, str) and model_name.rsplit('/', 1)[-1].startswith('claude-sonnet'):
				llm_screenshot_size = (1400, 850)
				logger.info('🖼️  Auto-configured LLM screenshot size for Claude Sonnet: 1400x850')

		if page_extraction_llm is None:
			page_extraction_llm = llm
		if judge_llm is None:
			judge_llm = llm
		if review_llm is None:
			review_llm = judge_llm
		if available_file_paths is None:
			available_file_paths = []

		# Set timeout based on model name if not explicitly provided
		if llm_timeout is None:

			def _get_model_timeout(llm_model: BaseChatModel) -> int:
				"""Determine timeout based on model name"""
				model_name = getattr(llm_model, 'model', '').lower()
				if 'gemini' in model_name:
					if '3-pro' in model_name:
						return 90
					return 75
				elif 'groq' in model_name:
					return 30
				elif 'o3' in model_name or 'claude' in model_name or 'sonnet' in model_name or 'deepseek' in model_name:
					return 90
				else:
					return 75  # Default timeout

			llm_timeout = _get_model_timeout(llm)

		self.id = task_id or uuid7str()
		self.task_id: str = self.id
		self.session_id: str = uuid7str()

		base_profile = browser_profile or DEFAULT_BROWSER_PROFILE
		if base_profile is DEFAULT_BROWSER_PROFILE:
			base_profile = base_profile.model_copy()
		if self._qa_scope is not None:
			base_profile = base_profile.model_copy(update={'qa_root_url': self._qa_scope.root_url})
		if demo_mode is not None and base_profile.demo_mode != demo_mode:
			base_profile = base_profile.model_copy(update={'demo_mode': demo_mode})
		browser_profile = base_profile

		# Handle browser vs browser_session parameter (browser takes precedence)
		if browser and browser_session:
			raise ValueError('Cannot specify both "browser" and "browser_session" parameters. Use "browser" for the cleaner API.')
		browser_session = browser or browser_session

		if browser_session is not None and demo_mode is not None and browser_session.browser_profile.demo_mode != demo_mode:
			browser_session.browser_profile = browser_session.browser_profile.model_copy(update={'demo_mode': demo_mode})

		self.browser_session = browser_session or BrowserSession(
			browser_profile=browser_profile,
			id=uuid7str()[:-4] + self.id[-4:],  # re-use the same 4-char suffix so they show up together in logs
		)
		if self._qa_scope is not None and self.browser_session.browser_profile.qa_root_url != self._qa_scope.root_url:
			self.browser_session.browser_profile = self.browser_session.browser_profile.model_copy(
				update={'qa_root_url': self._qa_scope.root_url}
			)
		if self._qa_scope is not None:
			from browser_use.browser.watchdogs.security_watchdog import SecurityWatchdog

			policy = SecurityWatchdog(event_bus=self.browser_session.event_bus, browser_session=self.browser_session)
			if not policy._is_url_allowed(self._qa_scope.root_url):
				self._qa_scope_error = (
					'QA root URL conflicts with the caller allowed_domains, prohibited_domains, or IP-address policy'
				)

		self._demo_mode_enabled: bool = bool(self.browser_profile.demo_mode) if self.browser_session else False
		if self._demo_mode_enabled and getattr(self.browser_profile, 'headless', False):
			self.logger.warning(
				'Demo mode is enabled but the browser is headless=True; set headless=False to view the in-browser panel.'
			)

		# Initialize available file paths as direct attribute
		self.available_file_paths = available_file_paths

		# Set up tools first (needed to detect output_model_schema)
		if tools is not None:
			self.tools = tools
		elif controller is not None:
			self.tools = controller
		else:
			# Exclude screenshot tool when use_vision is not auto
			exclude_actions = ['screenshot'] if use_vision != 'auto' else []
			self.tools = Tools(exclude_actions=exclude_actions, display_files_in_done_text=display_files_in_done_text)

		# Enforce screenshot exclusion when use_vision != 'auto', even if user passed custom tools
		if use_vision != 'auto':
			self.tools.exclude_action('screenshot')

		# Enable coordinate clicking for models that support it
		model_name = getattr(llm, 'model', '').lower()
		supports_coordinate_clicking = any(
			pattern in model_name
			for pattern in ['claude-sonnet-4', 'claude-opus-4', 'claude-fable-5', 'gemini-3-pro', 'browser-use/']
		)
		if supports_coordinate_clicking:
			self.tools.set_coordinate_clicking(True)

		# Handle skills vs skill_ids parameter (skills takes precedence)
		if skills and skill_ids:
			raise ValueError('Cannot specify both "skills" and "skill_ids" parameters. Use "skills" for the cleaner API.')
		skill_ids = skills or skill_ids

		# Skills integration - use injected service or create from skill_ids
		self.skill_service = None
		self._skills_registered = False
		if skill_service is not None:
			self.skill_service = skill_service
		elif skill_ids:
			from browser_use.skills import SkillService

			self.skill_service = SkillService(skill_ids=skill_ids)

		# Structured output - use explicit param or detect from tools
		tools_output_model = self.tools.get_output_model()
		if output_model_schema is not None and tools_output_model is not None:
			# Both provided - warn if they differ
			if output_model_schema is not tools_output_model:
				logger.warning(
					f'output_model_schema ({output_model_schema.__name__}) differs from Tools output_model '
					f'({tools_output_model.__name__}). Using Agent output_model_schema.'
				)
		elif output_model_schema is None and tools_output_model is not None:
			# Only tools has it - use that (cast is safe: both are BaseModel subclasses)
			output_model_schema = cast(type[AgentStructuredOutput], tools_output_model)
		if output_model_schema is not None:
			logger.warning('output_model_schema is ignored by the QA Agent; read the fixed history.qa_result schema')
		output_model_schema = None
		self.output_model_schema = None

		# QA owns the terminal result schema. The executor can only signal that the
		# current business step is ready for an independent judgement.
		# Valid QA specifications cannot self-complete through done(success=...). Keep
		# the legacy schema only for INVALID_SPEC objects, whose run() exits before
		# any executor/browser call; this preserves harmless inspection helpers.
		if self._qa_scope is not None:
			self.tools.exclude_action('done')

		@self.tools.action(
			'Finish the current business test step and submit objective observations for independent judgement. '
			'Do not decide pass or fail yourself. This action must be called alone.',
			param_model=FinishTestStepAction,
			terminates_sequence=True,
		)
		async def finish_test_step(params: FinishTestStepAction):
			report = params.model_dump(mode='json')
			return ActionResult(
				is_done=True,
				success=None,
				extracted_content=params.actual_result,
				long_term_memory=f'Business step submitted for independent QA judgement: {params.actual_result}',
				metadata={'qa_finish_test_step': report},
			)

		# Per-page extract uses a schema only when the caller explicitly asks for one.
		# It must NOT inherit output_model_schema: that describes the final task result
		# (e.g. {summary, step_results}), which is the wrong shape for a single-page
		# extraction and, on the browser-use gateway, routes extract into the agent
		# action protocol and breaks it.
		self.extraction_schema = extraction_schema

		# Core components - task enhancement now has access to output_model_schema from tools
		self.task = self._enhance_task_with_schema(task, output_model_schema)
		self.llm = llm
		self.judge_llm = judge_llm
		self.review_llm = review_llm

		# Fallback LLM configuration
		self._fallback_llm: BaseChatModel | None = fallback_llm
		self._using_fallback_llm: bool = False
		self._original_llm: BaseChatModel = llm  # Store original for reference
		self.directly_open_url = directly_open_url
		self.include_recent_events = include_recent_events
		self._url_shortening_limit = _url_shortening_limit

		self.sensitive_data = sensitive_data

		self.sample_images = sample_images

		if isinstance(message_compaction, bool):
			message_compaction = MessageCompactionSettings(enabled=message_compaction)

		self.settings = AgentSettings(
			use_vision=use_vision,
			vision_detail_level=vision_detail_level,
			save_conversation_path=save_conversation_path,
			save_conversation_path_encoding=save_conversation_path_encoding,
			# A low-level tool/model failure immediately forces a step-boundary report.
			# Only the independent judge may authorize up to
			# max_agent_retries_per_step safe business-step replays.
			max_failures=1,
			override_system_message=override_system_message,
			extend_system_message=extend_system_message,
			generate_gif=generate_gif,
			include_attributes=include_attributes,
			max_actions_per_step=max_actions_per_step,
			use_thinking=use_thinking,
			flash_mode=flash_mode,
			max_history_items=max_history_items,
			page_extraction_llm=page_extraction_llm,
			calculate_cost=calculate_cost,
			include_tool_call_examples=include_tool_call_examples,
			llm_timeout=llm_timeout,
			step_timeout=step_timeout,
			final_response_after_failure=final_response_after_failure,
			use_judge=use_judge,
			ground_truth=ground_truth,
			enable_planning=enable_planning,
			planning_replan_on_stall=planning_replan_on_stall,
			planning_exploration_limit=planning_exploration_limit,
			loop_detection_window=loop_detection_window,
			loop_detection_enabled=loop_detection_enabled,
			message_compaction=message_compaction,
			max_clickable_elements_length=max_clickable_elements_length,
			max_agent_retries_per_step=max_agent_retries_per_step,
			reuse_compiled_test_case=reuse_compiled_test_case,
			reuse_login_state=reuse_login_state,
			qa_step_limit=qa_step_limit,
		)

		# Token cost service
		self.token_cost_service = TokenCost(include_cost=calculate_cost, pricing_url=pricing_url)
		self.token_cost_service.register_llm(llm)
		self.token_cost_service.register_llm(page_extraction_llm)
		self.token_cost_service.register_llm(judge_llm)
		self.token_cost_service.register_llm(review_llm)
		if self.settings.message_compaction and self.settings.message_compaction.compaction_llm:
			self.token_cost_service.register_llm(self.settings.message_compaction.compaction_llm)

		# Store signal handler setting (not part of AgentSettings as it's runtime behavior)
		self.enable_signal_handler = enable_signal_handler

		# Initialize state
		self.state = injected_agent_state or AgentState()

		# Configure loop detector window size from settings
		self.state.loop_detector.window_size = self.settings.loop_detection_window

		# Initialize history
		self.history = AgentHistoryList(history=[], usage=None)

		# Initialize agent directory
		import time

		timestamp = int(time.time())
		base_tmp = Path(tempfile.gettempdir())
		self.agent_directory = base_tmp / f'browser_use_agent_{self.id}_{timestamp}'

		# Initialize file system and screenshot service
		self._set_file_system(file_system_path)
		self._set_screenshot_service()

		# Action setup
		self._setup_action_models()
		self._set_browser_use_version_and_source(source)

		initial_url = None

		# The QA runner owns root navigation. Keeping it out of initial_actions avoids
		# racing compilation/exploration and guarantees one formal root navigation.
		if self.directly_open_url and not self.state.follow_up_task and not initial_actions:
			# Preserve constructor-level URL discovery metadata for compatibility,
			# while run() still returns INVALID_SPEC before browser startup unless an
			# explicit HTTP(S) URL produced a valid QA scope.
			initial_url = self._qa_scope.root_url if self._qa_scope is not None else self._extract_start_url(task)
			if initial_url:
				self.logger.info(f'🔗 Found QA root URL in task: {initial_url}; navigation is runner-managed.')

		self.initial_url = initial_url

		self.initial_actions = self._convert_initial_actions(initial_actions) if initial_actions else None
		# Verify we can connect to the model
		self._verify_and_setup_llm()

		# TODO: move this logic to the LLMs
		# Handle users trying to use use_vision=True with DeepSeek models
		if 'deepseek' in self.llm.model.lower():
			self.logger.warning('⚠️ DeepSeek models do not support use_vision=True yet. Setting use_vision=False for now...')
			self.settings.use_vision = False

		# Handle users trying to use use_vision=True with XAI models that don't support it
		# grok-3 variants and grok-code don't support vision; grok-2 and grok-4 do
		model_lower = self.llm.model.lower()
		if 'grok-3' in model_lower or 'grok-code' in model_lower:
			self.logger.warning('⚠️ This XAI model does not support use_vision=True yet. Setting use_vision=False for now...')
			self.settings.use_vision = False

		logger.debug(
			f'{" +vision" if self.settings.use_vision else ""}'
			f' extraction_model={self.settings.page_extraction_llm.model if self.settings.page_extraction_llm else "Unknown"}'
			f'{" +file_system" if self.file_system else ""}'
		)

		# Store llm_screenshot_size in browser_session so tools can access it
		self.browser_session.llm_screenshot_size = llm_screenshot_size

		# Check if LLM is ChatAnthropic instance
		from browser_use.llm.anthropic.chat import ChatAnthropic

		is_anthropic = isinstance(self.llm, ChatAnthropic)

		# Check if model is a browser-use fine-tuned model (uses simplified prompts)
		is_browser_use_model = 'browser-use/' in self.llm.model.lower()

		# Initialize message manager with state
		# Initial system prompt with all actions - will be updated during each step
		qa_system_extension = """
<web_ui_qa_mode>
This agent executes Web UI QA business steps. The latest QA STEP message is the only current objective.
Do not optimize for completing an overall task and do not decide whether the SUT passed or failed.
Perform only the actions needed for the current business step. Verify the resulting page using objective browser evidence.
When enough evidence exists, call finish_test_step as a single action. Report what is actually visible, not a success opinion.
Never navigate outside the configured QA root domain. Do not use search engines or external research.
For submit, delete, payment, or other irreversible actions, do not repeat the action when its side effect is uncertain; set side_effect_uncertain instead.
</web_ui_qa_mode>
"""
		resolved_system_extension = qa_system_extension
		if extend_system_message:
			resolved_system_extension = f'{extend_system_message}\n{qa_system_extension}'

		self._message_manager = MessageManager(
			# The compiler receives the full natural-language specification. The
			# executor receives only the current business step later in
			# _prepare_qa_case(), preventing it from racing ahead or self-judging the case.
			task='Wait for the QA runner to provide the current compiled business step.',
			system_message=SystemPrompt(
				max_actions_per_step=self.settings.max_actions_per_step,
				override_system_message=override_system_message,
				extend_system_message=resolved_system_extension,
				use_thinking=self.settings.use_thinking,
				flash_mode=self.settings.flash_mode,
				is_anthropic=is_anthropic,
				is_browser_use_model=is_browser_use_model,
				model_name=self.llm.model,
			).get_system_message(),
			file_system=self.file_system,
			state=self.state.message_manager_state,
			use_thinking=self.settings.use_thinking,
			# Settings that were previously in MessageManagerSettings
			include_attributes=self.settings.include_attributes,
			sensitive_data=sensitive_data,
			max_history_items=self.settings.max_history_items,
			vision_detail_level=self.settings.vision_detail_level,
			include_tool_call_examples=self.settings.include_tool_call_examples,
			include_recent_events=self.include_recent_events,
			sample_images=self.sample_images,
			llm_screenshot_size=llm_screenshot_size,
			max_clickable_elements_length=self.settings.max_clickable_elements_length,
		)

		if self.sensitive_data:
			# Check if sensitive_data has domain-specific credentials
			has_domain_specific_credentials = any(isinstance(v, dict) for v in self.sensitive_data.values())

			# If no allowed_domains are configured, show a security warning
			if not self.browser_profile.allowed_domains:
				self.logger.warning(
					'⚠️ Agent(sensitive_data=••••••••) was provided but Browser(allowed_domains=[...]) is not locked down! ⚠️\n'
					'          ☠️ If the agent visits a malicious website and encounters a prompt-injection attack, your sensitive_data may be exposed!\n\n'
					'   \n'
				)

			# If we're using domain-specific credentials, validate domain patterns
			elif has_domain_specific_credentials:
				# For domain-specific format, ensure all domain patterns are included in allowed_domains
				domain_patterns = [k for k, v in self.sensitive_data.items() if isinstance(v, dict)]

				# Validate each domain pattern against allowed_domains
				for domain_pattern in domain_patterns:
					is_allowed = False
					for allowed_domain in self.browser_profile.allowed_domains:
						# Special cases that don't require URL matching
						if domain_pattern == allowed_domain or allowed_domain == '*':
							is_allowed = True
							break

						# Need to create example URLs to compare the patterns
						# Extract the domain parts, ignoring scheme
						pattern_domain = domain_pattern.split('://')[-1] if '://' in domain_pattern else domain_pattern
						allowed_domain_part = allowed_domain.split('://')[-1] if '://' in allowed_domain else allowed_domain

						# Check if pattern is covered by an allowed domain
						# Example: "google.com" is covered by "*.google.com"
						if pattern_domain == allowed_domain_part or (
							allowed_domain_part.startswith('*.')
							and (
								pattern_domain == allowed_domain_part[2:]
								or pattern_domain.endswith('.' + allowed_domain_part[2:])
							)
						):
							is_allowed = True
							break

					if not is_allowed:
						self.logger.warning(
							f'⚠️ Domain pattern "{domain_pattern}" in sensitive_data is not covered by any pattern in allowed_domains={self.browser_profile.allowed_domains}\n'
							f'   This may be a security risk as credentials could be used on unintended domains.'
						)

		# Callbacks
		self.register_new_step_callback = register_new_step_callback
		self.register_done_callback = register_done_callback
		self.register_qa_plan_callback = register_qa_plan_callback
		self.register_should_stop_callback = register_should_stop_callback
		self.register_external_agent_status_raise_error_callback = register_external_agent_status_raise_error_callback

		# Telemetry
		self.telemetry = ProductTelemetry()

		# Event bus with WAL persistence
		# Default to ~/.config/browseruse/events/{agent_session_id}.jsonl
		# wal_path = CONFIG.BROWSER_USE_CONFIG_DIR / 'events' / f'{self.session_id}.jsonl'
		self.eventbus = EventBus(name=f'Agent_{str(self.id)[-4:]}')

		if self.settings.save_conversation_path:
			self.settings.save_conversation_path = Path(self.settings.save_conversation_path).expanduser().resolve()
			self.logger.info(f'💬 Saving conversation to {_log_pretty_path(self.settings.save_conversation_path)}')

		# Initialize download tracking
		assert self.browser_session is not None, 'BrowserSession is not set up'
		self.has_downloads_path = self.browser_session.browser_profile.downloads_path is not None
		if self.has_downloads_path:
			self._last_known_downloads: list[str] = []
			self.logger.debug('📁 Initialized download tracking for agent')

		# Event-based pause control (kept out of AgentState for serialization)
		self._external_pause_event = asyncio.Event()
		self._external_pause_event.set()

	def _enhance_task_with_schema(self, task: str, output_model_schema: type[AgentStructuredOutput] | None) -> str:
		"""Enhance task description with output schema information if provided."""
		if output_model_schema is None:
			return task

		try:
			schema = output_model_schema.model_json_schema()
			import json

			schema_json = json.dumps(schema, indent=2)

			enhancement = f'\nExpected output format: {output_model_schema.__name__}\n{schema_json}'
			return task + enhancement
		except Exception as e:
			self.logger.debug(f'Could not parse output schema: {e}')

		return task

	@property
	def logger(self) -> logging.Logger:
		"""Get instance-specific logger with task ID in the name"""
		# logger may be called in __init__ so we don't assume self.* attributes have been initialized
		_task_id = task_id[-4:] if (task_id := getattr(self, 'task_id', None)) else '----'
		_browser_session_id = browser_session.id[-4:] if (browser_session := getattr(self, 'browser_session', None)) else '----'
		_current_target_id = (
			browser_session.agent_focus_target_id[-2:]
			if (browser_session := getattr(self, 'browser_session', None)) and browser_session.agent_focus_target_id
			else '--'
		)
		return logging.getLogger(f'browser_use.Agent🅰 {_task_id} ⇢ 🅑 {_browser_session_id} 🅣 {_current_target_id}')

	@property
	def browser_profile(self) -> BrowserProfile:
		assert self.browser_session is not None, 'BrowserSession is not set up'
		return self.browser_session.browser_profile

	@property
	def is_using_fallback_llm(self) -> bool:
		"""Check if the agent is currently using the fallback LLM."""
		return self._using_fallback_llm

	@property
	def current_llm_model(self) -> str:
		"""Get the model name of the currently active LLM."""
		return self.llm.model if hasattr(self.llm, 'model') else 'unknown'

	async def _check_and_update_downloads(self, context: str = '') -> None:
		"""Check for new downloads and update available file paths."""
		if not self.has_downloads_path:
			return

		assert self.browser_session is not None, 'BrowserSession is not set up'

		try:
			current_downloads = self.browser_session.downloaded_files
			if current_downloads != self._last_known_downloads:
				self._update_available_file_paths(current_downloads)
				self._last_known_downloads = current_downloads
				if context:
					self.logger.debug(f'📁 {context}: Updated available files')
		except Exception as e:
			error_context = f' {context}' if context else ''
			self.logger.debug(f'📁 Failed to check for downloads{error_context}: {type(e).__name__}: {e}')

	def _update_available_file_paths(self, downloads: list[str]) -> None:
		"""Update available_file_paths with downloaded files."""
		if not self.has_downloads_path:
			return

		current_files = set(self.available_file_paths or [])
		new_files = set(downloads) - current_files

		if new_files:
			self.available_file_paths = list(current_files | new_files)

			self.logger.info(
				f'📁 Added {len(new_files)} downloaded files to available_file_paths (total: {len(self.available_file_paths)} files)'
			)
			for file_path in new_files:
				self.logger.info(f'📄 New file available: {file_path}')
		else:
			self.logger.debug(f'📁 No new downloads detected (tracking {len(current_files)} files)')

	def _set_file_system(self, file_system_path: str | None = None) -> None:
		# Check for conflicting parameters
		if self.state.file_system_state and file_system_path:
			raise ValueError(
				'Cannot provide both file_system_state (from agent state) and file_system_path. '
				'Either restore from existing state or create new file system at specified path, not both.'
			)

		# Check if we should restore from existing state first
		if self.state.file_system_state:
			try:
				# Restore file system from state at the exact same location
				self.file_system = FileSystem.from_state(self.state.file_system_state)
				# The parent directory of base_dir is the original file_system_path
				self.file_system_path = str(self.file_system.base_dir)
				self.logger.debug(f'💾 File system restored from state to: {self.file_system_path}')
				return
			except Exception as e:
				self.logger.error(f'💾 Failed to restore file system from state: {e}')
				raise e

		# Initialize new file system
		try:
			if file_system_path:
				self.file_system = FileSystem(file_system_path)
				self.file_system_path = file_system_path
			else:
				# Use the agent directory for file system
				self.file_system = FileSystem(self.agent_directory)
				self.file_system_path = str(self.agent_directory)
		except Exception as e:
			self.logger.error(f'💾 Failed to initialize file system: {e}.')
			raise e

		# Save file system state to agent state
		self.state.file_system_state = self.file_system.get_state()

		self.logger.debug(f'💾 File system path: {self.file_system_path}')

	def _set_screenshot_service(self) -> None:
		"""Initialize screenshot service using agent directory"""
		try:
			from browser_use.screenshots.service import ScreenshotService

			self.screenshot_service = ScreenshotService(self.agent_directory)
			self.logger.debug(f'📸 Screenshot service initialized in: {self.agent_directory}/screenshots')
		except Exception as e:
			self.logger.error(f'📸 Failed to initialize screenshot service: {e}.')
			raise e

	def save_file_system_state(self) -> None:
		"""Save current file system state to agent state"""
		if self.file_system:
			self.state.file_system_state = self.file_system.get_state()
		else:
			self.logger.error('💾 File system is not set up. Cannot save state.')
			raise ValueError('File system is not set up. Cannot save state.')

	def _set_browser_use_version_and_source(self, source_override: str | None = None) -> None:
		"""Get the version from pyproject.toml and determine the source of the browser-use package"""
		# Use the helper function for version detection
		version = get_browser_use_version()

		# Determine source
		try:
			package_root = Path(__file__).parent.parent.parent
			repo_files = ['.git', 'README.md', 'docs', 'examples']
			if all(Path(package_root / file).exists() for file in repo_files):
				source = 'git'
			else:
				source = 'pip'
		except Exception as e:
			self.logger.debug(f'Error determining source: {e}')
			source = 'unknown'

		if source_override is not None:
			source = source_override
		# self.logger.debug(f'Version: {version}, Source: {source}')  # moved later to _log_agent_run so that people are more likely to include it in copy-pasted support ticket logs
		self.version = version
		self.source = source

	def _setup_action_models(self) -> None:
		"""Setup dynamic action models from tools registry"""
		# Initially only include actions with no filters
		self.ActionModel = self.tools.registry.create_action_model()
		# Create output model with the dynamic actions
		if self.settings.flash_mode:
			self.AgentOutput = AgentOutput.type_with_custom_actions_flash_mode(self.ActionModel)
		elif self.settings.use_thinking:
			self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)
		else:
			self.AgentOutput = AgentOutput.type_with_custom_actions_no_thinking(self.ActionModel)

		# Used to force a QA step boundary when the low-level step budget is reached.
		self.DoneActionModel = self.tools.registry.create_action_model(include_actions=[self._completion_action_name])
		if self.settings.flash_mode:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions_flash_mode(self.DoneActionModel)
		elif self.settings.use_thinking:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions(self.DoneActionModel)
		else:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions_no_thinking(self.DoneActionModel)

	def _get_skill_slug(self, skill: 'Skill', all_skills: list['Skill']) -> str:
		"""Generate a clean slug from skill title for action names

		Converts title to lowercase, removes special characters, replaces spaces with underscores.
		Adds UUID suffix if there are duplicate slugs.

		Args:
			skill: The skill to get slug for
			all_skills: List of all skills to check for duplicates

		Returns:
			Slug like "cloned_github_stars_tracker" or "get_weather_data_a1b2" if duplicate

		Examples:
			"[Cloned] Github Stars Tracker" -> "cloned_github_stars_tracker"
			"Get Weather Data" -> "get_weather_data"
		"""
		import re

		# Remove special characters and convert to lowercase
		slug = re.sub(r'[^\w\s]', '', skill.title.lower())
		# Replace whitespace and hyphens with underscores
		slug = re.sub(r'[\s\-]+', '_', slug)
		# Remove leading/trailing underscores
		slug = slug.strip('_')

		# Check for duplicates and add UUID suffix if needed
		same_slug_count = sum(
			1 for s in all_skills if re.sub(r'[\s\-]+', '_', re.sub(r'[^\w\s]', '', s.title.lower()).strip('_')) == slug
		)
		if same_slug_count > 1:
			return f'{slug}_{skill.id[:4]}'
		else:
			return slug

	async def _register_skills_as_actions(self) -> None:
		"""Register each skill as a separate action using slug as action name"""
		if not self.skill_service or self._skills_registered:
			return

		self.logger.info('🔧 Registering skill actions...')

		# Fetch all skills (auto-initializes if needed)
		skills = await self.skill_service.get_all_skills()

		if not skills:
			self.logger.warning('No skills loaded from SkillService')
			return

		# Register each skill as its own action
		for skill in skills:
			slug = self._get_skill_slug(skill, skills)
			param_model = skill.parameters_pydantic(exclude_cookies=True)

			# Create description with skill title in quotes
			description = f'{skill.description} (Skill: "{skill.title}")'

			# Create handler for this specific skill
			def make_skill_handler(skill_id: str):
				async def skill_handler(params: BaseModel) -> ActionResult:
					"""Execute a specific skill"""
					assert self.skill_service is not None, 'SkillService not initialized'

					# Convert parameters to dict
					if isinstance(params, BaseModel):
						skill_params = params.model_dump()
					elif isinstance(params, dict):
						skill_params = params
					else:
						return ActionResult(extracted_content=None, error=f'Invalid parameters type: {type(params)}')

					# Get cookies from browser
					_cookies = await self.browser_session.cookies()

					try:
						result = await self.skill_service.execute_skill(
							skill_id=skill_id, parameters=skill_params, cookies=_cookies
						)

						if result.success:
							return ActionResult(
								extracted_content=str(result.result) if result.result else None,
								error=None,
							)
						else:
							return ActionResult(extracted_content=None, error=result.error or 'Skill execution failed')
					except Exception as e:
						# Check if it's a MissingCookieException
						if type(e).__name__ == 'MissingCookieException':
							# Format: "Missing cookies (name): description"
							cookie_name = getattr(e, 'cookie_name', 'unknown')
							cookie_description = getattr(e, 'cookie_description', str(e))
							error_msg = f'Missing cookies ({cookie_name}): {cookie_description}'
							return ActionResult(extracted_content=None, error=error_msg)
						return ActionResult(extracted_content=None, error=f'Skill execution error: {type(e).__name__}: {e}')

				return skill_handler

			# Create the handler for this skill
			handler = make_skill_handler(skill.id)
			handler.__name__ = slug

			# Register the action with the slug as the action name
			self.tools.registry.action(description=description, param_model=param_model)(handler)

		# Mark as registered
		self._skills_registered = True

		# Rebuild action models to include the new skill actions
		self._setup_action_models()

		# Reconvert initial actions with the new ActionModel type if they exist
		if self.initial_actions:
			# Convert back to dict form first
			initial_actions_dict = []
			for action in self.initial_actions:
				action_dump = action.model_dump(exclude_unset=True)
				initial_actions_dict.append(action_dump)
			# Reconvert using new ActionModel
			self.initial_actions = self._convert_initial_actions(initial_actions_dict)

		self.logger.info(f'✓ Registered {len(skills)} skill actions')

	async def _get_unavailable_skills_info(self) -> str:
		"""Get information about skills that are unavailable due to missing cookies

		Returns:
			Formatted string describing unavailable skills and how to make them available
		"""
		if not self.skill_service:
			return ''

		try:
			# Get all skills
			skills = await self.skill_service.get_all_skills()
			if not skills:
				return ''

			# Get current cookies
			current_cookies = await self.browser_session.cookies()
			cookie_dict = {cookie['name']: cookie['value'] for cookie in current_cookies}

			# Check each skill for missing required cookies
			unavailable_skills: list[dict[str, Any]] = []

			for skill in skills:
				# Get cookie parameters for this skill
				cookie_params = [p for p in skill.parameters if p.type == 'cookie']

				if not cookie_params:
					# No cookies needed, skip
					continue

				# Check for missing required cookies
				missing_cookies: list[dict[str, str]] = []
				for cookie_param in cookie_params:
					is_required = cookie_param.required if cookie_param.required is not None else True

					if is_required and cookie_param.name not in cookie_dict:
						missing_cookies.append(
							{'name': cookie_param.name, 'description': cookie_param.description or 'No description provided'}
						)

				if missing_cookies:
					unavailable_skills.append(
						{
							'id': skill.id,
							'title': skill.title,
							'description': skill.description,
							'missing_cookies': missing_cookies,
						}
					)

			if not unavailable_skills:
				return ''

			# Format the unavailable skills info with slugs
			lines = ['Unavailable Skills (missing required cookies):']
			for skill_info in unavailable_skills:
				# Get the full skill object to use the slug helper
				skill_obj = next((s for s in skills if s.id == skill_info['id']), None)
				slug = self._get_skill_slug(skill_obj, skills) if skill_obj else skill_info['title']
				title = skill_info['title']

				lines.append(f'\n  • {slug} ("{title}")')
				lines.append(f'    Description: {skill_info["description"]}')
				lines.append('    Missing cookies:')
				for cookie in skill_info['missing_cookies']:
					lines.append(f'      - {cookie["name"]}: {cookie["description"]}')

			return '\n'.join(lines)

		except Exception as e:
			self.logger.error(f'Error getting unavailable skills info: {type(e).__name__}: {e}')
			return ''

	def add_new_task(self, new_task: str) -> None:
		"""Add a new task to the agent, keeping the same task_id as tasks are continuous"""
		previous_run_completed = self.history.qa_result is not None
		reuse_cached_case = bool(
			self.settings.reuse_compiled_test_case
			and self._qa_test_case is not None
			and self._qa_compiled_task is not None
			and new_task.strip() == self._qa_compiled_task.strip()
		)
		try:
			new_scope = QATaskCompiler.resolve_scope(new_task)
		except ValueError as exc:
			self._qa_scope_error = str(exc)
			new_scope = None
		if self._qa_scope is not None and new_scope is not None:
			if new_scope.registrable_domain != self._qa_scope.registrable_domain:
				raise ValueError('A follow-up QA task cannot change registrable domain; create a new Agent instead')
		self._qa_original_task = new_task
		if new_scope is not None:
			self._qa_scope = new_scope
			self._qa_scope_error = None
			self.browser_session.browser_profile = self.browser_session.browser_profile.model_copy(
				update={'qa_root_url': new_scope.root_url}
			)
			from browser_use.browser.watchdogs.security_watchdog import SecurityWatchdog

			policy = SecurityWatchdog(event_bus=self.browser_session.event_bus, browser_session=self.browser_session)
			if not policy._is_url_allowed(new_scope.root_url):
				self._qa_scope_error = (
					'QA root URL conflicts with the caller allowed_domains, prohibited_domains, or IP-address policy'
				)
		if not reuse_cached_case:
			self._qa_test_case = None
			self._qa_compiled_task = None
			self._qa_replay_history = {}
		self._qa_test_case_draft = None
		self._qa_step_results = []
		self._qa_current_step_index = 0
		self._qa_step_retry_count = 0
		self._qa_current_attempt_receipts = []
		self._qa_evidence_cursor = None
		self._qa_phase_timings = {}
		self._qa_done_callback_called = False
		self._qa_llm_call_count = 0
		self._qa_requested_mode = 'ai'
		self._qa_effective_mode = 'ai'
		self._qa_precondition_results = []
		self._qa_warnings = []
		self._qa_running_preconditions = False
		self._qa_precondition_step = None
		self._qa_run_id = uuid7str()
		self._qa_cleanup_results = []
		self._qa_running_cleanup = False
		self._qa_cleanup_index = 0
		self._qa_pending_business_outcome = None
		self._qa_replay_allow_llm_fallback = True
		self._qa_replay_fallback_active = False
		self._qa_replay_fallback_completed = False
		self.history.qa_result = None
		# Simply delegate to message manager - no need for new task_id or events
		# The task continues with new instructions, it doesn't end and start a new one
		self.task = new_task
		message = (
			'Reusing the validated QA specification. Wait for the current QA STEP.'
			if reuse_cached_case
			else 'A new QA specification is being compiled. Wait for the current QA STEP.'
		)
		if previous_run_completed:
			self._reset_qa_execution_message_context()
		self._message_manager.add_new_task(message)
		# Mark as follow-up task and recreate eventbus (gets shut down after each run)
		self.state.follow_up_task = True
		# Reset control flags so agent can continue
		self.state.stopped = False
		self.state.paused = False
		self.state.n_steps = 1
		self.state.consecutive_failures = 0
		self.state.last_result = None
		self.state.last_model_output = None
		if self.state.session_initialized or previous_run_completed:
			agent_id_suffix = str(self.id)[-4:].replace('-', '_')
			if agent_id_suffix and agent_id_suffix[0].isdigit():
				agent_id_suffix = 'a' + agent_id_suffix
			self.eventbus = EventBus(name=f'Agent_{agent_id_suffix}')

	def _reset_qa_execution_message_context(self) -> None:
		"""Exclude a completed run's executor transcript from the next independent QA run."""

		current_system_message = self._message_manager.state.history.system_message or self._message_manager.system_prompt
		fresh_state = type(self.state.message_manager_state)()
		fresh_state.history.system_message = current_system_message
		self.state.message_manager_state = fresh_state
		self._message_manager.state = fresh_state
		self._message_manager.task = 'Wait for the QA runner to provide the current compiled business step.'
		self._message_manager.last_input_messages = []
		self._message_manager.last_state_message_text = None
		self._message_manager.sensitive_data_description = ''

	def _record_qa_phase_timing(self, phase: str, elapsed_seconds: float) -> None:
		"""Accumulate a named wall-clock measurement and update an attached result."""

		self._qa_phase_timings[phase] = self._qa_phase_timings.get(phase, 0.0) + max(elapsed_seconds, 0.0)
		self._sync_qa_phase_timings()

	def _record_qa_llm_call(self) -> None:
		"""Count one provider invocation performed by a QA compiler, executor, or Judge."""

		self._qa_llm_call_count += 1

	def _sync_qa_phase_timings(self) -> None:
		"""Expose current runner timings through the strongly typed QA result."""

		if self.history.qa_result is None:
			return
		self.history.qa_result.phase_timings = [
			QAPhaseTiming(phase=phase, elapsed_seconds=elapsed) for phase, elapsed in self._qa_phase_timings.items()
		]

	def save_qa_bundle(self, path: str | Path) -> QABundle:
		"""Persist the compiled case, redacted evidence, and replay traces as a v2 bundle."""

		if self.history.qa_result is None:
			raise ValueError('Cannot save a QA bundle before the Agent has produced qa_result')
		action_history = {
			step_id: [item.model_dump(sensitive_data=self.sensitive_data) for item in history]
			for step_id, history in self._qa_replay_history.items()
		}
		redacted_result = self._redacted_qa_result_for_persistence(self.history.qa_result)
		return QABundle.save(
			path,
			task=self._qa_original_task,
			ground_truth=self.settings.ground_truth,
			run_result=redacted_result,
			action_history=action_history,
		)

	def _redacted_qa_result_for_persistence(self, result: QARunResult) -> QARunResult:
		"""Apply a final defense-in-depth redaction before reports or bundles are written."""

		serialized = result.model_dump_json()
		sensitive_values = collect_sensitive_data_values(self.sensitive_data)
		if sensitive_values:
			serialized = redact_sensitive_string(serialized, sensitive_values)
		return QARunResult.model_validate_json(serialized)

	def save_qa_report(
		self,
		path: str | Path,
		*,
		format: Literal['json', 'junit', 'html'] | None = None,
	) -> Path:
		"""Write the current QA result as JSON, JUnit XML, or self-contained HTML."""

		if self.history.qa_result is None:
			raise ValueError('Cannot save a QA report before the Agent has produced qa_result')
		from browser_use.qa.reporting import write_qa_html_report, write_qa_json_report, write_qa_junit_report

		report_path = Path(path)
		selected_format = format
		if selected_format is None:
			selected_format = {'.json': 'json', '.xml': 'junit', '.html': 'html', '.htm': 'html'}.get(
				report_path.suffix.casefold()
			)
		if selected_format is None:
			raise ValueError('Report format must be json, junit, or html (or use a recognized file extension)')
		writer = {
			'json': write_qa_json_report,
			'junit': write_qa_junit_report,
			'html': write_qa_html_report,
		}[selected_format]
		return writer(self._redacted_qa_result_for_persistence(self.history.qa_result), report_path)

	def _load_qa_bundle_for_replay(self, bundle: QABundle | str | Path) -> QARunResult:
		"""Validate a bundle against this Agent and restore its cross-process trace."""

		loaded = bundle if isinstance(bundle, QABundle) else QABundle.load(bundle)
		if loaded.manifest.task_hash != qa_content_hash(self._qa_original_task):
			raise ValueError('QA bundle Task does not match this Agent Task')
		if loaded.manifest.ground_truth_hash != qa_content_hash(self.settings.ground_truth):
			raise ValueError('QA bundle ground_truth does not match this Agent')
		if self._qa_scope is None:
			raise ValueError('QA bundle replay requires a valid Task navigation scope')
		if loaded.manifest.root_url != self._qa_scope.root_url:
			raise ValueError('QA bundle root URL does not match this Agent Task')
		if loaded.manifest.registrable_domain != self._qa_scope.registrable_domain:
			raise ValueError('QA bundle registrable domain does not match this Agent Task')
		if loaded.run_result.legacy_imported:
			raise ValueError('Legacy imported QA results cannot be used as a v2 replay baseline')
		if loaded.run_result.status != QARunStatus.PASSED or not loaded.run_result.has_reliable_verdict:
			raise ValueError('QA bundle replay requires a reliable PASSED baseline')

		self._qa_test_case = loaded.test_case.model_copy(deep=True)
		self._qa_compiled_task = self._qa_original_task
		self._qa_replay_history = {}
		for step_id, serialized_items in loaded.action_history.items():
			loaded_history = AgentHistoryList.load_from_dict({'history': serialized_items}, self.AgentOutput)
			self._qa_replay_history[step_id] = [item.model_copy(deep=True) for item in loaded_history.history]
		return loaded.run_result.model_copy(deep=True)

	@contextmanager
	def _measure_qa_phase(self, phase: str) -> Iterator[None]:
		"""Measure one synchronous or asynchronous QA runner block."""

		started_at = time.perf_counter()
		try:
			yield
		finally:
			self._record_qa_phase_timing(phase, time.perf_counter() - started_at)

	async def rerun(
		self,
		max_steps: int = 500,
		on_step_start: AgentHookFunc | None = None,
		on_step_end: AgentHookFunc | None = None,
		mode: Literal['ai', 'replay'] = 'ai',
		bundle: QABundle | str | Path | None = None,
		allow_llm_fallback: bool = True,
	) -> AgentHistoryList[AgentStructuredOutput]:
		"""Repeat a QA Task using AI or audited replay with optional AI drift repair."""

		if mode not in {'ai', 'replay'}:
			raise ValueError("rerun mode must be 'ai' or 'replay'")
		baseline_result = self.history.qa_result.model_copy(deep=True) if self.history.qa_result is not None else None
		if bundle is not None:
			if mode != 'replay':
				raise ValueError('bundle is only supported with mode="replay"')
			baseline_result = self._load_qa_bundle_for_replay(bundle)
		self.add_new_task(self._qa_original_task)
		if mode == 'replay':
			self._qa_requested_mode = 'replay'
			self._qa_effective_mode = 'replay'
			self._qa_replay_allow_llm_fallback = allow_llm_fallback
			return await self._run_qa_replay(
				baseline_result,
				max_steps=max_steps,
				on_step_start=on_step_start,
				on_step_end=on_step_end,
			)
		self._qa_requested_mode = 'ai'
		self._qa_effective_mode = 'ai'
		return await self.run(max_steps=max_steps, on_step_start=on_step_start, on_step_end=on_step_end)

	def _ensure_browser_session_can_restart(self) -> None:
		"""Restore BrowserSession lifecycle handlers cleared by the previous closed run."""

		start_handlers = self.browser_session.event_bus.handlers.get(BrowserStartEvent.__name__, [])
		if not start_handlers:
			self.browser_session.model_post_init(None)

	@staticmethod
	def _qa_replay_action_name(action: ActionModel) -> str:
		action_data = action.model_dump(exclude_unset=True)
		return next(iter(action_data), '')

	def _prepare_qa_replay_history_item(self, history_item: AgentHistory) -> tuple[AgentHistory | None, list[str], str | None]:
		"""Prepare one saved action batch for fast, model-free execution."""

		if history_item.model_output is None:
			return None, [], None
		kept_actions: list[ActionModel] = []
		kept_elements: list[DOMInteractedElement | None] = []
		action_names: list[str] = []
		interacted_elements = history_item.state.interacted_element or []
		for index, action in enumerate(history_item.model_output.action):
			action_name = self._qa_replay_action_name(action)
			if action_name == self._completion_action_name:
				continue
			if action_name == 'extract':
				return None, [], 'extract actions require an LLM and cannot be used in strict replay mode'
			replay_action = action.model_copy(deep=True)
			if action_name == 'wait':
				# Subsequent target-aware waits replace the executor's conservative fixed delay.
				# Keep the action for trace compatibility, but make Browser Use's wait action a no-op
				# (it subtracts one second before sleeping).
				action_data = replay_action.model_dump(exclude_unset=True)
				wait_params = action_data.get(action_name)
				if isinstance(wait_params, dict) and isinstance(wait_params.get('seconds'), int):
					wait_params['seconds'] = min(wait_params['seconds'], 1)
					replay_action = type(action).model_validate(action_data)
			kept_actions.append(replay_action)
			action_names.append(action_name)
			kept_elements.append(interacted_elements[index] if index < len(interacted_elements) else None)
		if not kept_actions:
			return None, [], None

		replay_item = history_item.model_copy(deep=True)
		assert replay_item.model_output is not None
		replay_item.model_output.action = [action.model_copy(deep=True) for action in kept_actions]
		replay_item.state.interacted_element = kept_elements
		return replay_item, action_names, None

	@staticmethod
	def _qa_replay_is_interaction(action_name: str) -> bool:
		return action_name not in {'wait', 'screenshot', 'scroll', 'read_state'}

	def _append_qa_replay_terminal_history(self, snapshot: BrowserEvidenceSnapshot, message: str) -> None:
		"""Expose the replay's final browser state through the existing history interface."""

		self.history.add_item(
			AgentHistory(
				model_output=None,
				result=[ActionResult(is_done=False, extracted_content=message, long_term_memory=message)],
				state=BrowserStateHistory(
					url=snapshot.url,
					title=snapshot.title,
					tabs=[],
					interacted_element=[],
					screenshot_path=snapshot.screenshot_path,
				),
				metadata=None,
			)
		)

	def _judge_qa_replay_step(
		self,
		*,
		step: WebUITestStep,
		baseline: QAStepResult,
		evidence: StepEvidence,
		last_interaction_completed: bool,
		had_interactions: bool,
	) -> StepJudgement:
		"""Apply the first reliable run's typed assertions without invoking a model."""

		assertions = baseline.judgement.replay_assertions if baseline.judgement is not None else []
		evidence_ids = evidence.after.evidence_ids
		if not assertions:
			return StepJudgement(
				action_status=ActionCompletionStatus.UNCERTAIN,
				expectation_status=ExpectationStatus.NOT_OBSERVABLE,
				status=QAStepStatus.INCONCLUSIVE,
				failure_origin=FailureOrigin.UNKNOWN,
				reasoning='The reliable first run produced no deterministic replay assertions.',
				actual_result='The step cannot be revalidated without an LLM.',
			)

		assertion_results = [(assertion, replay_assertion_matches(assertion, evidence.after)) for assertion in assertions]
		failed = [assertion for assertion, matched in assertion_results if not matched]
		assertion_evidence = [
			f'{assertion.kind.value}({assertion.value!r})={"MET" if matched else "NOT_MET"}'
			for assertion, matched in assertion_results
		]
		if not failed and (last_interaction_completed or not had_interactions):
			return StepJudgement(
				action_status=ActionCompletionStatus.COMPLETED,
				expectation_status=ExpectationStatus.MET,
				status=QAStepStatus.PASSED,
				failure_origin=FailureOrigin.NONE,
				reasoning='The saved action completed and every deterministic assertion from the reliable first run is met.',
				actual_result='All deterministic replay assertions are met.',
				evidence=assertion_evidence,
				evidence_ids=evidence_ids,
				confidence=1.0,
				replay_assertions=assertions,
			)
		if not failed:
			return StepJudgement(
				action_status=ActionCompletionStatus.NOT_COMPLETED,
				expectation_status=ExpectationStatus.MET,
				status=QAStepStatus.AGENT_FAILED,
				failure_origin=FailureOrigin.AGENT,
				reasoning='The expected state is present, but the saved final interaction could not be confirmed.',
				actual_result='Deterministic replay did not confirm the intended business action.',
				evidence=assertion_evidence,
			)

		if any(assertion.kind.value.startswith('dom_') and not evidence.after.dom_summary.strip() for assertion in failed):
			return StepJudgement(
				action_status=(
					ActionCompletionStatus.COMPLETED if last_interaction_completed else ActionCompletionStatus.UNCERTAIN
				),
				expectation_status=ExpectationStatus.NOT_OBSERVABLE,
				status=QAStepStatus.INCONCLUSIVE,
				failure_origin=FailureOrigin.UNKNOWN,
				reasoning='DOM evidence is empty, so failed DOM assertions are not a reliable product verdict.',
				actual_result='Expected DOM evidence could not be observed.',
				evidence=assertion_evidence,
			)
		if not last_interaction_completed and had_interactions:
			return StepJudgement(
				action_status=ActionCompletionStatus.NOT_COMPLETED,
				expectation_status=ExpectationStatus.NOT_OBSERVABLE,
				status=QAStepStatus.AGENT_FAILED,
				failure_origin=FailureOrigin.AGENT,
				reasoning='The saved final business interaction could not be replayed.',
				actual_result='Deterministic action replay failed before the expected state was established.',
				evidence=assertion_evidence,
			)
		if step.expectation_source == ExpectationSource.HEURISTIC:
			return StepJudgement(
				action_status=ActionCompletionStatus.COMPLETED,
				expectation_status=ExpectationStatus.NOT_OBSERVABLE,
				status=QAStepStatus.INCONCLUSIVE,
				failure_origin=FailureOrigin.UNKNOWN,
				reasoning='A heuristic expectation cannot establish a SUT defect in model-free replay.',
				actual_result='One or more heuristic replay assertions are not met.',
				evidence=assertion_evidence,
			)
		return StepJudgement(
			action_status=ActionCompletionStatus.COMPLETED,
			expectation_status=ExpectationStatus.NOT_MET,
			status=QAStepStatus.SUT_FAILED,
			failure_origin=FailureOrigin.SUT,
			reasoning='The saved business action completed, but reliable deterministic assertions are not met.',
			actual_result='One or more deterministic expected-result assertions are not met.',
			evidence=assertion_evidence,
			evidence_ids=evidence_ids,
			confidence=1.0,
			failure_code=FailureCode.SUT_EXPECTATION_MISMATCH,
		)

	async def _run_ai_fallback_for_replay_step(
		self,
		*,
		step_index: int,
		fallback_reason: str,
		max_steps: int,
		on_step_start: AgentHookFunc | None,
		on_step_end: AgentHookFunc | None,
	) -> bool:
		"""Let AI repair only the drifted replay step, then return control to replay."""

		assert self._qa_test_case is not None
		step = self._qa_test_case.steps[step_index]
		self._qa_effective_mode = 'ai_fallback'
		self._qa_warnings.append(f'AI fallback for {step.step_id}: {fallback_reason}')
		self.logger.warning(f'↪️ Replay drift at {step.step_id}; invoking AI only for this step: {fallback_reason}')
		if self._qa_step_results and self._qa_step_results[-1].step.step_id == step.step_id:
			self._qa_step_results.pop()
		self._qa_current_step_index = step_index
		self._qa_step_retry_count = 0
		self._qa_before_snapshot = await self._capture_qa_snapshot(f'{step.step_id}_ai_fallback_before', include_screenshot=True)
		self._qa_evidence_cursor = self._qa_evidence_monitor.cursor() if self._qa_evidence_monitor else None
		self._qa_step_history_start = len(self.history.history)
		self._qa_phase = 'ai_fallback'
		self._qa_replay_fallback_active = True
		self._qa_replay_fallback_completed = False
		self._reset_qa_execution_message_context()
		self._message_manager.add_new_task(self._qa_step_prompt(step))
		for fallback_step_number in range(max_steps):
			step_info = AgentStepInfo(step_number=fallback_step_number, max_steps=max_steps)
			is_done = await self._execute_step(
				fallback_step_number,
				max_steps,
				step_info,
				on_step_start,
				on_step_end,
			)
			if not is_done:
				continue
			terminal = await self._handle_qa_step_boundary()
			if self._qa_replay_fallback_completed:
				self._qa_phase = 'replay'
				return True
			if terminal:
				return False
		self._qa_replay_fallback_active = False
		self._finalize_qa_result(
			status=QARunStatus.AGENT_FAILED,
			failure_origin=FailureOrigin.AGENT,
			failure_code=FailureCode.AGENT_STEP_BUDGET,
			stopped_at_step=step.step_id,
			summary=f'AGENT_FAILED at {step.step_id}: AI fallback exhausted its step budget.',
		)
		return False

	async def _run_qa_replay(
		self,
		baseline_result: QARunResult | None,
		*,
		max_steps: int,
		on_step_start: AgentHookFunc | None,
		on_step_end: AgentHookFunc | None,
	) -> AgentHistoryList[AgentStructuredOutput]:
		"""Execute and judge a previously passed QA case without any LLM calls."""

		replay_started_at = time.perf_counter()
		self._session_start_time = time.time()
		self._task_start_time = self._session_start_time
		self._qa_phase = 'replay'
		self._qa_step_results = []
		self._qa_phase_timings = {}
		last_snapshot = BrowserEvidenceSnapshot(url=self._qa_scope.root_url if self._qa_scope else '')
		try:
			if (
				baseline_result is None
				or baseline_result.legacy_imported
				or baseline_result.status != QARunStatus.PASSED
				or not baseline_result.has_reliable_verdict
				or baseline_result.test_case is None
			):
				self._append_qa_replay_terminal_history(last_snapshot, 'No reliable PASSED baseline is available for replay.')
				self._finalize_qa_result(
					status=QARunStatus.INCONCLUSIVE,
					failure_origin=FailureOrigin.UNKNOWN,
					summary='INCONCLUSIVE: deterministic replay requires a reliable PASSED first run.',
				)
				return self.history
			if self._qa_scope is None or self._qa_test_case is None:
				self._append_qa_replay_terminal_history(last_snapshot, 'The cached QA specification is unavailable.')
				self._finalize_qa_result(
					status=QARunStatus.INCONCLUSIVE,
					failure_origin=FailureOrigin.UNKNOWN,
					summary='INCONCLUSIVE: deterministic replay requires a cached QA specification.',
				)
				return self.history

			baseline_steps = {result.step.step_id: result for result in baseline_result.step_results}
			await self._notify_qa_plan_callback(self._qa_plan_snapshot_from_case(self._qa_test_case))
			self.logger.info('⚙️ Starting deterministic QA replay with zero LLM calls.')
			self._ensure_browser_session_can_restart()
			with self._measure_qa_phase('browser_start'):
				await self.browser_session.start()
			with self._measure_qa_phase('login_state_restore'):
				await self._restore_cached_qa_login_state()
			self._qa_evidence_monitor = QAEvidenceMonitor(self.browser_session, self._qa_scope)
			with self._measure_qa_phase('evidence_monitor_start'):
				await self._qa_evidence_monitor.start()
			with self._measure_qa_phase('root_navigation'):
				await self.browser_session.navigate_to(self._qa_scope.root_url, new_tab=False)
				await self._wait_for_qa_page_stability()

			for step_index, step in enumerate(self._qa_test_case.steps):
				baseline_step = baseline_steps.get(step.step_id)
				replay_history = self._qa_replay_history.get(step.step_id, [])
				if baseline_step is None or not replay_history:
					last_snapshot = await self._capture_qa_snapshot(
						f'{step.step_id}_replay_unavailable', include_screenshot=False
					)
					judgement = StepJudgement(
						action_status=ActionCompletionStatus.UNCERTAIN,
						expectation_status=ExpectationStatus.NOT_OBSERVABLE,
						status=QAStepStatus.INCONCLUSIVE,
						failure_origin=FailureOrigin.UNKNOWN,
						reasoning='No reliable in-memory action trace is available for this step.',
						actual_result='The step cannot be replayed without an LLM.',
					)
					evidence = StepEvidence(before=last_snapshot, after=last_snapshot)
					self._qa_step_results.append(
						QAStepResult(step=step, status=judgement.status, judgement=judgement, evidence=evidence)
					)
					if self._qa_replay_allow_llm_fallback:
						fallback_passed = await self._run_ai_fallback_for_replay_step(
							step_index=step_index,
							fallback_reason=judgement.reasoning,
							max_steps=max_steps,
							on_step_start=on_step_start,
							on_step_end=on_step_end,
						)
						if fallback_passed:
							continue
						break
					self._append_qa_replay_terminal_history(last_snapshot, judgement.actual_result)
					self._finalize_qa_result(
						status=QARunStatus.INCONCLUSIVE,
						failure_origin=FailureOrigin.UNKNOWN,
						stopped_at_step=step.step_id,
						summary=f'INCONCLUSIVE at {step.step_id}: {judgement.actual_result}',
					)
					break

				self._qa_evidence_cursor = self._qa_evidence_monitor.cursor()
				with self._measure_qa_phase(f'{step.step_id}.before_evidence'):
					before = await self._capture_qa_lightweight_snapshot()
				prepared_items: list[tuple[AgentHistory, list[str]]] = []
				unsupported_reason: str | None = None
				for history_item in replay_history:
					prepared_item, action_names, error = self._prepare_qa_replay_history_item(history_item)
					if error:
						unsupported_reason = error
						break
					if prepared_item is not None:
						prepared_items.append((prepared_item, action_names))

				had_interactions = any(
					self._qa_replay_is_interaction(action_name)
					for _, action_names in prepared_items
					for action_name in action_names
				)
				interaction_total = sum(
					self._qa_replay_is_interaction(action_name)
					for _, action_names in prepared_items
					for action_name in action_names
				)
				interaction_index = 0
				last_interaction_completed = False
				action_results: list[dict[str, Any]] = []
				if unsupported_reason:
					action_results.append({'error': unsupported_reason})
				else:
					for prepared_item, action_names in prepared_items:
						try:
							await asyncio.sleep(min(self.browser_profile.wait_between_actions, 0.5))
							with self._measure_qa_phase(f'{step.step_id}.target_wait'):
								initial_replay_state = await self.browser_session.get_browser_state_summary(
									include_screenshot=False
								)
								if self._qa_replay_authentication_batch_is_obsolete(prepared_item, initial_replay_state):
									interaction_index += sum(
										self._qa_replay_is_interaction(action_name) for action_name in action_names
									)
									action_results.append(
										{
											'warning': 'Skipped saved authentication actions because reused login state is already authenticated.',
											'skipped_actions': action_names,
										}
									)
									continue
								replay_state = await self._wait_for_history_targets(
									prepared_item,
									timeout=15.0,
									poll_interval=0.5,
									initial_state=initial_replay_state,
								)
							with self._measure_qa_phase(f'{step.step_id}.action_execution'):
								results = await self._execute_history_step(
									prepared_item,
									delay=0,
									ai_step_llm=None,
									wait_for_elements=False,
									browser_state_summary=replay_state,
								)
							action_results.extend(
								result.model_dump(exclude_none=True, mode='json', exclude={'images'}) for result in results
							)
							for result_index, action_name in enumerate(action_names):
								if not self._qa_replay_is_interaction(action_name):
									continue
								interaction_index += 1
								if interaction_index == interaction_total:
									last_interaction_completed = (
										result_index < len(results) and results[result_index].error is None
									)
						except Exception as exc:
							action_results.append({'error': f'{type(exc).__name__}: {exc}'})
							interaction_index += sum(self._qa_replay_is_interaction(action_name) for action_name in action_names)

				with self._measure_qa_phase(f'{step.step_id}.after_evidence'):
					last_snapshot = await self._capture_qa_snapshot(f'{step.step_id}_replay_after', include_screenshot=False)
				replay_receipt = ActionReceipt(
					status=(
						ActionCompletionStatus.COMPLETED
						if last_interaction_completed or not had_interactions
						else ActionCompletionStatus.NOT_COMPLETED
					),
					operation_kind=step.operation_kind,
					tool_succeeded=last_interaction_completed or not had_interactions,
					state_changed=before.url != last_snapshot.url or before.dom_summary != last_snapshot.dom_summary,
					side_effect_uncertain=step.side_effect_level == SideEffectLevel.IRREVERSIBLE
					and not last_interaction_completed,
					action_names=[name for _, names in prepared_items for name in names],
					evidence_ids=last_snapshot.evidence_ids,
					reasoning='Receipt derived from deterministic tool results and replayed browser state.',
				)
				evidence = StepEvidence(
					before=before,
					after=last_snapshot,
					action_results=action_results,
					action_receipt=replay_receipt,
					evidence_quality=(EvidenceQuality.STRONG if last_snapshot.dom_summary.strip() else EvidenceQuality.WEAK),
				)
				if unsupported_reason:
					judgement = StepJudgement(
						action_status=ActionCompletionStatus.UNCERTAIN,
						expectation_status=ExpectationStatus.NOT_OBSERVABLE,
						status=QAStepStatus.INCONCLUSIVE,
						failure_origin=FailureOrigin.UNKNOWN,
						reasoning=unsupported_reason,
						actual_result='The saved step contains an action that is not model-free.',
					)
				else:
					judgement = self._judge_qa_replay_step(
						step=step,
						baseline=baseline_step,
						evidence=evidence,
						last_interaction_completed=last_interaction_completed,
						had_interactions=had_interactions,
					)
				review = None
				if judgement.status == QAStepStatus.SUT_FAILED:
					review = ReviewRecord(
						primary_status=QAStepStatus.SUT_FAILED,
						secondary_status=QAStepStatus.SUT_FAILED,
						agreed=True,
						reviewer_model='deterministic-replay-validator',
						reason='A separate deterministic assertion pass confirmed the same mismatch.',
						evidence_ids=judgement.evidence_ids,
					)
				self._qa_step_results.append(
					QAStepResult(
						step=step,
						status=judgement.status,
						judgement=judgement,
						review=review,
						evidence=evidence,
						attempt_receipts=[replay_receipt],
					)
				)
				if judgement.status != QAStepStatus.PASSED:
					may_fallback = (
						self._qa_replay_allow_llm_fallback
						and judgement.status in {QAStepStatus.AGENT_FAILED, QAStepStatus.INCONCLUSIVE}
						and not replay_receipt.side_effect_uncertain
					)
					if may_fallback:
						fallback_passed = await self._run_ai_fallback_for_replay_step(
							step_index=step_index,
							fallback_reason=judgement.reasoning,
							max_steps=max_steps,
							on_step_start=on_step_start,
							on_step_end=on_step_end,
						)
						if fallback_passed:
							continue
						break
					run_status, failure_origin = self._qa_run_status_for_step(judgement.status)
					self._append_qa_replay_terminal_history(last_snapshot, judgement.actual_result)
					self._finalize_qa_result(
						status=run_status,
						failure_origin=failure_origin,
						stopped_at_step=step.step_id,
						summary=f'{run_status.value} at {step.step_id}: {judgement.actual_result}',
					)
					break
			else:
				self._append_qa_replay_terminal_history(
					last_snapshot,
					f'PASSED: all {len(self._qa_test_case.steps)} steps passed deterministic replay.',
				)
				self._finalize_qa_result(
					status=QARunStatus.PASSED,
					failure_origin=FailureOrigin.NONE,
					summary=f'PASSED: all {len(self._qa_test_case.steps)} steps passed deterministic replay.',
				)
		except Exception as exc:
			status, origin = self._qa_runtime_failure_status(exc, phase='replay')
			self._append_qa_replay_terminal_history(last_snapshot, f'{status.value}: {type(exc).__name__}: {exc}')
			self._finalize_qa_result(
				status=status,
				failure_origin=origin,
				failure_code=self._qa_runtime_failure_code(exc, phase='replay'),
				summary=f'{status.value} during deterministic replay: {type(exc).__name__}: {exc}',
			)
		finally:
			with self._measure_qa_phase('result_callbacks'):
				self.history.usage = await self.token_cost_service.get_usage_summary()
				await self._notify_qa_done_callback()
			with self._measure_qa_phase('login_state_cache'):
				await self._cache_qa_login_state()
			await self.token_cost_service.log_usage_summary()
			with self._measure_qa_phase('shutdown'):
				await self.eventbus.stop(clear=True, timeout=_get_timeout('TIMEOUT_AgentEventBusStop', 3.0))
				await self.close()
			self._record_qa_phase_timing('replay_total', time.perf_counter() - replay_started_at)
			self._sync_qa_phase_timings()
		return self.history

	async def _check_stop_or_pause(self) -> None:
		"""Check if the agent should stop or pause, and handle accordingly."""

		# Check new should_stop_callback - sets stopped state cleanly without raising
		if self.register_should_stop_callback:
			if await self.register_should_stop_callback():
				self.logger.info('External callback requested stop')
				self.state.stopped = True
				raise InterruptedError

		if self.register_external_agent_status_raise_error_callback:
			if await self.register_external_agent_status_raise_error_callback():
				raise InterruptedError

		if self.state.stopped:
			raise InterruptedError

		if self.state.paused:
			raise InterruptedError

	@observe(name='agent.step', ignore_output=True, ignore_input=True)
	@time_execution_async('--step')
	async def step(self, step_info: AgentStepInfo | None = None) -> None:
		"""Execute one step of the task"""
		# Initialize timing first, before any exceptions can occur

		self.step_start_time = time.time()

		browser_state_summary = None

		try:
			if self.browser_session:
				try:
					captcha_wait = await self.browser_session.wait_if_captcha_solving()
					if captcha_wait and captcha_wait.waited:
						# Reset step timing to exclude the captcha wait from step duration metrics
						self.step_start_time = time.time()
						duration_s = captcha_wait.duration_ms / 1000
						outcome = captcha_wait.result  # 'success' | 'failed' | 'timeout'
						msg = f'Waited {duration_s:.1f}s for {captcha_wait.vendor} CAPTCHA to be solved. Result: {outcome}.'
						self.logger.info(f'🔒 {msg}')
						# Inject the outcome so the LLM sees what happened
						captcha_result = ActionResult(long_term_memory=msg)
						if self.state.last_result:
							self.state.last_result.append(captcha_result)
						else:
							self.state.last_result = [captcha_result]
				except Exception as e:
					self.logger.warning(f'Phase 0 captcha wait failed (non-fatal): {e}')

			# Phase 1: Prepare context and timing
			browser_state_summary = await self._prepare_context(step_info)

			# Clear previous step state after context preparation (which needs
			# them for the "previous action result" prompt) but before the LLM
			# call, so a timeout during _get_next_action or _execute_actions
			# won't leave stale data from the previous step.
			self.state.last_model_output = None
			self.state.last_result = None

			# Phase 2: Get model output and execute actions
			await self._get_next_action(browser_state_summary)
			await self._execute_actions()

			# Phase 3: Post-processing
			await self._post_process()

		except Exception as e:
			# Handle ALL exceptions in one place
			await self._handle_step_error(e)

		finally:
			await self._finalize(browser_state_summary)

	async def _prepare_context(self, step_info: AgentStepInfo | None = None) -> BrowserStateSummary:
		"""Prepare the context for the step: browser state, action models, page actions"""
		# step_start_time is now set in step() method

		assert self.browser_session is not None, 'BrowserSession is not set up'

		self.logger.debug(f'🌐 Step {self.state.n_steps}: Getting browser state...')
		# Always take screenshots for all steps
		self.logger.debug('📸 Requesting browser state with include_screenshot=True')
		browser_state_summary = await self.browser_session.get_browser_state_summary(
			include_screenshot=True,  # always capture even if use_vision=False so that cloud sync is useful (it's fast now anyway)
			include_recent_events=self.include_recent_events,
		)
		if browser_state_summary.screenshot:
			self.logger.debug(f'📸 Got browser state WITH screenshot, length: {len(browser_state_summary.screenshot)}')
		else:
			self.logger.debug('📸 Got browser state WITHOUT screenshot')

		# Check for new downloads after getting browser state (catches PDF auto-downloads and previous step downloads)
		await self._check_and_update_downloads(f'Step {self.state.n_steps}: after getting browser state')

		self._log_step_context(browser_state_summary)
		await self._check_stop_or_pause()

		# Update action models with page-specific actions
		self.logger.debug(f'📝 Step {self.state.n_steps}: Updating action models...')
		await self._update_action_models_for_page(browser_state_summary.url)

		# Get page-specific filtered actions
		page_filtered_actions = self.tools.registry.get_prompt_description(browser_state_summary.url)

		# Page-specific actions will be included directly in the browser_state message
		self.logger.debug(f'💬 Step {self.state.n_steps}: Creating state messages for context...')

		# Get unavailable skills info if skills service is enabled
		unavailable_skills_info = None
		if self.skill_service is not None:
			unavailable_skills_info = await self._get_unavailable_skills_info()

		# Render plan description for injection into agent context
		plan_description = self._render_plan_description()

		self._message_manager.prepare_step_state(
			browser_state_summary=browser_state_summary,
			model_output=self.state.last_model_output,
			result=self.state.last_result,
			step_info=step_info,
			sensitive_data=self.sensitive_data,
		)

		await self._maybe_compact_messages(step_info)

		self._message_manager.create_state_messages(
			browser_state_summary=browser_state_summary,
			model_output=self.state.last_model_output,
			result=self.state.last_result,
			step_info=step_info,
			use_vision=self.settings.use_vision,
			page_filtered_actions=page_filtered_actions if page_filtered_actions else None,
			sensitive_data=self.sensitive_data,
			available_file_paths=self.available_file_paths,  # Always pass current available_file_paths
			unavailable_skills_info=unavailable_skills_info,
			plan_description=plan_description,
			skip_state_update=True,
		)

		await self._inject_budget_warning(step_info)
		self._inject_replan_nudge()
		self._inject_exploration_nudge()
		self._update_loop_detector_page_state(browser_state_summary)
		self._inject_loop_detection_nudge()
		await self._force_done_after_last_step(step_info)
		await self._force_done_after_failure()
		return browser_state_summary

	async def _maybe_compact_messages(self, step_info: AgentStepInfo | None = None) -> None:
		"""Optionally compact message history to keep prompts small."""
		settings = self.settings.message_compaction
		if not settings or not settings.enabled:
			return

		compaction_llm = settings.compaction_llm or self.settings.page_extraction_llm or self.llm
		await self._message_manager.maybe_compact_messages(
			llm=compaction_llm,
			settings=settings,
			step_info=step_info,
		)

	@observe_debug(ignore_input=True, name='get_next_action')
	async def _get_next_action(self, browser_state_summary: BrowserStateSummary) -> None:
		"""Execute LLM interaction with retry logic and handle callbacks"""
		input_messages = self._message_manager.get_messages()
		self.logger.debug(
			f'🤖 Step {self.state.n_steps}: Calling LLM with {len(input_messages)} messages (model: {self.llm.model})...'
		)

		try:
			self._qa_llm_call_count += 1
			model_output = await asyncio.wait_for(
				self._get_model_output_with_retry(input_messages), timeout=self.settings.llm_timeout
			)
		except TimeoutError:

			@observe(name='_llm_call_timed_out_with_input')
			async def _log_model_input_to_lmnr(input_messages: list[BaseMessage]) -> None:
				"""Log the model input"""
				pass

			await _log_model_input_to_lmnr(input_messages)

			raise TimeoutError(
				f'LLM call timed out after {self.settings.llm_timeout} seconds. Keep your thinking and output short.'
			)

		self.state.last_model_output = model_output

		# Check again for paused/stopped state after getting model output
		await self._check_stop_or_pause()

		# Handle callbacks and conversation saving
		await self._handle_post_llm_processing(browser_state_summary, input_messages)

		# check again if Ctrl+C was pressed before we commit the output to history
		await self._check_stop_or_pause()

	async def _execute_actions(self) -> None:
		"""Execute the actions from model output"""
		if self.state.last_model_output is None:
			raise ValueError('No model output to execute actions from')

		result = await self.multi_act(self.state.last_model_output.action)
		self.state.last_result = result

	async def _post_process(self) -> None:
		"""Handle post-action processing like download tracking and result logging"""
		assert self.browser_session is not None, 'BrowserSession is not set up'

		# Check for new downloads after executing actions
		await self._check_and_update_downloads('after executing actions')

		# Update plan state from model output
		if self.state.last_model_output is not None:
			self._update_plan_from_model_output(self.state.last_model_output)

		# Record executed actions for loop detection
		self._update_loop_detector_actions()

		# check for action errors - only count single-action steps toward consecutive failures;
		# multi-action steps with errors are handled by loop detection and replan nudges instead
		if self.state.last_result and len(self.state.last_result) == 1 and self.state.last_result[-1].error:
			self.state.consecutive_failures += 1
			self.logger.debug(f'🔄 Step {self.state.n_steps}: Consecutive failures: {self.state.consecutive_failures}')
			return

		if self.state.consecutive_failures > 0:
			self.state.consecutive_failures = 0
			self.logger.debug(f'🔄 Step {self.state.n_steps}: Consecutive failures reset to: {self.state.consecutive_failures}')

		# Log completion results
		if self.state.last_result and len(self.state.last_result) > 0 and self.state.last_result[-1].is_done:
			success = self.state.last_result[-1].success
			if success:
				# Green color for success
				self.logger.info(f'\n📄 \033[32m Final Result:\033[0m \n{self.state.last_result[-1].extracted_content}\n\n')
			else:
				# Red color for failure
				self.logger.info(f'\n📄 \033[31m Final Result:\033[0m \n{self.state.last_result[-1].extracted_content}\n\n')
			if self.state.last_result[-1].attachments:
				total_attachments = len(self.state.last_result[-1].attachments)
				for i, file_path in enumerate(self.state.last_result[-1].attachments):
					self.logger.info(f'👉 Attachment {i + 1 if total_attachments > 1 else ""}: {file_path}')

	async def _handle_step_error(self, error: Exception) -> None:
		"""Handle all types of errors that can occur during a step"""

		# Handle InterruptedError specially
		if isinstance(error, InterruptedError):
			error_msg = 'The agent was interrupted mid-step' + (f' - {str(error)}' if str(error) else '')
			# NOTE: This is not an error, it's a normal part of the execution when the user interrupts the agent
			self.logger.warning(f'{error_msg}')
			return

		# Handle browser closed/disconnected errors
		if self._is_connection_like_error(error):
			# If reconnection is in progress, wait for it instead of stopping
			if self.browser_session.is_reconnecting:
				wait_timeout = self.browser_session.RECONNECT_WAIT_TIMEOUT
				self.logger.warning(
					f'🔄 Connection error during reconnection, waiting up to {wait_timeout}s for reconnect: {error}'
				)
				try:
					await asyncio.wait_for(self.browser_session._reconnect_event.wait(), timeout=wait_timeout)
				except TimeoutError:
					pass

				# Check if reconnection succeeded
				if self.browser_session.is_cdp_connected:
					self.logger.info('🔄 Reconnection succeeded, retrying step...')
					self.state.last_result = [ActionResult(error=f'Connection lost and recovered: {error}')]
					return

			# Not reconnecting or reconnection failed — check if truly terminal
			if self._is_browser_closed_error(error):
				self.logger.warning(f'🛑 Browser closed or disconnected: {error}')
				self.state.stopped = True
				self._external_pause_event.set()
				return

		# Handle all other exceptions
		include_trace = self.logger.isEnabledFor(logging.DEBUG)
		error_msg = AgentError.format_error(error, include_trace=include_trace)
		max_total_failures = self.settings.max_failures + int(self.settings.final_response_after_failure)
		prefix = f'❌ Result failed {self.state.consecutive_failures + 1}/{max_total_failures} times: '
		self.state.consecutive_failures += 1

		# Use WARNING for partial failures, ERROR only when max failures reached
		is_final_failure = self.state.consecutive_failures >= max_total_failures
		log_level = logging.ERROR if is_final_failure else logging.WARNING

		if 'Could not parse response' in error_msg or 'tool_use_failed' in error_msg:
			# give model a hint how output should look like
			self.logger.log(log_level, f'Model: {self.llm.model} failed')
			self.logger.log(log_level, f'{prefix}{error_msg}')
		else:
			self.logger.log(log_level, f'{prefix}{error_msg}')

		await self._demo_mode_log(f'Step error: {error_msg}', 'error', {'step': self.state.n_steps})
		self.state.last_result = [ActionResult(error=error_msg)]
		return None

	def _is_connection_like_error(self, error: Exception) -> bool:
		"""Check if the error looks like a CDP/WebSocket connection failure.

		Unlike _is_browser_closed_error(), this does NOT check if the CDP client is None
		or if reconnection is in progress — it purely looks at the error signature.
		"""
		error_str = str(error).lower()
		return (
			isinstance(error, ConnectionError)
			or 'websocket connection closed' in error_str
			or 'connection closed' in error_str
			or 'browser has been closed' in error_str
			or 'browser closed' in error_str
			or 'no browser' in error_str
		)

	def _is_browser_closed_error(self, error: Exception) -> bool:
		"""Check if the browser has been closed or disconnected.

		Only returns True when the error itself is a CDP/WebSocket connection failure
		AND the CDP client is gone AND we're not actively reconnecting.
		Avoids false positives on unrelated errors (element not found, timeouts,
		parse errors) that happen to coincide with a transient None state during
		reconnects or resets.
		"""
		# During reconnection, don't treat connection errors as terminal
		if self.browser_session.is_reconnecting:
			return False

		error_str = str(error).lower()
		is_connection_error = (
			isinstance(error, ConnectionError)
			or 'websocket connection closed' in error_str
			or 'connection closed' in error_str
			or 'browser has been closed' in error_str
			or 'browser closed' in error_str
			or 'no browser' in error_str
		)
		return is_connection_error and self.browser_session._cdp_client_root is None

	async def _finalize(self, browser_state_summary: BrowserStateSummary | None) -> None:
		"""Finalize the step with history, logging, and events"""
		step_end_time = time.time()
		if not self.state.last_result:
			return

		if browser_state_summary:
			step_interval = None
			if len(self.history.history) > 0:
				last_history_item = self.history.history[-1]

				if last_history_item.metadata:
					previous_end_time = last_history_item.metadata.step_end_time
					previous_start_time = last_history_item.metadata.step_start_time
					step_interval = max(0, previous_end_time - previous_start_time)
			metadata = StepMetadata(
				step_number=self.state.n_steps,
				step_start_time=self.step_start_time,
				step_end_time=step_end_time,
				step_interval=step_interval,
			)

			# Use _make_history_item like main branch
			await self._make_history_item(
				self.state.last_model_output,
				browser_state_summary,
				self.state.last_result,
				metadata,
				state_message=self._message_manager.last_state_message_text,
			)

		# Log step completion summary
		summary_message = self._log_step_completion_summary(self.step_start_time, self.state.last_result)
		if summary_message:
			await self._demo_mode_log(summary_message, 'info', {'step': self.state.n_steps})

		# Save file system state after step completion
		self.save_file_system_state()

		# Emit both step created and executed events
		if browser_state_summary and self.state.last_model_output:
			# Extract key step data for the event
			actions_data = []
			if self.state.last_model_output.action:
				for action in self.state.last_model_output.action:
					action_dict = action.model_dump() if hasattr(action, 'model_dump') else {}
					actions_data.append(action_dict)

			# Emit CreateAgentStepEvent
			step_event = CreateAgentStepEvent.from_agent_step(
				self,
				self.state.last_model_output,
				self.state.last_result,
				actions_data,
				browser_state_summary,
			)
			self.eventbus.dispatch(step_event)

		# Increment step counter after step is fully completed
		self.state.n_steps += 1

	def _update_plan_from_model_output(self, model_output: AgentOutput) -> None:
		"""Update the plan state from model output fields (current_plan_item, plan_update)."""
		if not self.settings.enable_planning:
			return

		# If model provided a new plan via plan_update, replace the current plan
		if model_output.plan_update is not None:
			self.state.plan = [PlanItem(text=step_text) for step_text in model_output.plan_update]
			self.state.current_plan_item_index = 0
			self.state.plan_generation_step = self.state.n_steps
			if self.state.plan:
				self.state.plan[0].status = 'current'
			self.logger.info(
				f'📋 Plan {"updated" if self.state.plan_generation_step else "created"} with {len(self.state.plan)} steps'
			)
			return

		# If model provided a step index update, advance the plan
		if model_output.current_plan_item is not None and self.state.plan is not None:
			new_idx = model_output.current_plan_item
			# Clamp to valid range
			new_idx = max(0, min(new_idx, len(self.state.plan) - 1))
			old_idx = self.state.current_plan_item_index

			# Mark steps between old and new as done
			for i in range(old_idx, new_idx):
				if i < len(self.state.plan) and self.state.plan[i].status in ('current', 'pending'):
					self.state.plan[i].status = 'done'

			# Mark the new step as current
			if new_idx < len(self.state.plan):
				self.state.plan[new_idx].status = 'current'

			self.state.current_plan_item_index = new_idx

	def _render_plan_description(self) -> str | None:
		"""Render the current plan as a text description for injection into agent context."""
		if not self.settings.enable_planning or self.state.plan is None:
			return None

		markers = {'done': '[x]', 'current': '[>]', 'pending': '[ ]', 'skipped': '[-]'}
		lines = []
		for i, step in enumerate(self.state.plan):
			marker = markers.get(step.status, '[ ]')
			lines.append(f'{marker} {i}: {step.text}')
		return '\n'.join(lines)

	def _inject_replan_nudge(self) -> None:
		"""Inject a replan nudge when stall detection threshold is met."""
		if not self.settings.enable_planning or self.state.plan is None:
			return
		if self.settings.planning_replan_on_stall <= 0:
			return
		if self.state.consecutive_failures >= self.settings.planning_replan_on_stall:
			msg = (
				'REPLAN SUGGESTED: You have failed '
				f'{self.state.consecutive_failures} consecutive times. '
				'Your current plan may need revision. '
				'Output a new `plan_update` with revised steps to recover.'
			)
			self.logger.info(f'📋 Replan nudge injected after {self.state.consecutive_failures} consecutive failures')
			self._message_manager._add_context_message(UserMessage(content=msg))

	def _inject_exploration_nudge(self) -> None:
		"""Nudge the agent to create a plan or submit the current QA step."""
		if not self.settings.enable_planning or self.state.plan is not None:
			return
		if self.settings.planning_exploration_limit <= 0:
			return
		if self.state.n_steps >= self.settings.planning_exploration_limit:
			msg = (
				'PLANNING NUDGE: You have taken '
				f'{self.state.n_steps} steps without creating a plan. '
				'If the task is complex, output a `plan_update` with clear todo items now. '
				'If the current QA step is ready for judgement, call finish_test_step instead.'
			)
			self.logger.info(f'📋 Exploration nudge injected after {self.state.n_steps} steps without a plan')
			self._message_manager._add_context_message(UserMessage(content=msg))

	def _inject_loop_detection_nudge(self) -> None:
		"""Inject an escalating nudge when behavioral loops are detected."""
		if not self.settings.loop_detection_enabled:
			return
		nudge = self.state.loop_detector.get_nudge_message()
		if nudge:
			self.logger.info(
				f'🔁 Loop detection nudge injected (repetition={self.state.loop_detector.max_repetition_count}, '
				f'stagnation={self.state.loop_detector.consecutive_stagnant_pages})'
			)
			self._message_manager._add_context_message(UserMessage(content=nudge))

	def _update_loop_detector_actions(self) -> None:
		"""Record the actions from the latest step into the loop detector."""
		if not self.settings.loop_detection_enabled:
			return
		if self.state.last_model_output is None:
			return
		# Actions to exclude: wait always hashes identically (instant false positive),
		# done is terminal, go_back is navigation recovery
		_LOOP_EXEMPT_ACTIONS = {'wait', self._completion_action_name, 'go_back'}
		for action in self.state.last_model_output.action:
			action_data = action.model_dump(exclude_unset=True)
			action_name = next(iter(action_data.keys()), 'unknown')
			if action_name in _LOOP_EXEMPT_ACTIONS:
				continue
			params = action_data.get(action_name, {})
			if not isinstance(params, dict):
				params = {}
			self.state.loop_detector.record_action(action_name, params)

	def _update_loop_detector_page_state(self, browser_state_summary: BrowserStateSummary) -> None:
		"""Record the current page state for stagnation detection."""
		if not self.settings.loop_detection_enabled:
			return
		url = browser_state_summary.url or ''
		element_count = len(browser_state_summary.dom_state.selector_map) if browser_state_summary.dom_state else 0
		# Use the DOM text representation for fingerprinting
		dom_text = ''
		if browser_state_summary.dom_state:
			try:
				dom_text = browser_state_summary.dom_state.llm_representation()
			except Exception:
				dom_text = ''
		self.state.loop_detector.record_page_state(url, dom_text, element_count)

	async def _inject_budget_warning(self, step_info: AgentStepInfo | None = None) -> None:
		"""Inject a prominent budget warning when the agent has used >= 75% of its step budget.

		This gives the LLM advance notice to wrap up, save partial results, and call done
		rather than exhausting all steps with nothing saved.
		"""
		if step_info is None:
			return

		steps_used = step_info.step_number + 1  # Convert 0-indexed to 1-indexed
		budget_ratio = steps_used / step_info.max_steps

		if budget_ratio >= 0.75 and not step_info.is_last_step():
			steps_remaining = step_info.max_steps - steps_used
			pct = int(budget_ratio * 100)
			msg = (
				f'BUDGET WARNING: You have used {steps_used}/{step_info.max_steps} steps '
				f'({pct}%). {steps_remaining} steps remaining. '
				f'If the task cannot be completed in the remaining steps, prioritize: '
				f'(1) consolidate your results (save to files if the file system is in use), '
				f'(2) call finish_test_step with the objective evidence you have. '
				f'Partial results are far more valuable than exhausting all steps with nothing saved.'
			)
			self.logger.info(f'Step budget warning: {steps_used}/{step_info.max_steps} ({pct}%)')
			self._message_manager._add_context_message(UserMessage(content=msg))

	async def _force_done_after_last_step(self, step_info: AgentStepInfo | None = None) -> None:
		"""Handle special processing for the last step"""
		if step_info and step_info.is_last_step():
			# Add last step warning if needed
			msg = 'You reached max_steps. Your only available tool is finish_test_step.'
			msg += '\nReport only the objective state reached for the current QA business step.'
			msg += '\nSet side_effect_uncertain=true if an irreversible action may have happened.'
			self.logger.debug('Last step finishing up')
			self._message_manager._add_context_message(UserMessage(content=msg))
			self.AgentOutput = self.DoneAgentOutput

	async def _force_done_after_failure(self) -> None:
		"""Force done after failure"""
		# Create recovery message
		if self.state.consecutive_failures >= self.settings.max_failures and self.settings.final_response_after_failure:
			msg = f'You failed {self.settings.max_failures} times. Therefore we terminate the agent.'
			msg += '\nYour only available tool is finish_test_step. Report the objective state of the current QA step.'
			msg += '\nSet side_effect_uncertain=true if an irreversible action may have happened.'

			self.logger.debug('Force QA step submission because max_failures was reached.')
			self._message_manager._add_context_message(UserMessage(content=msg))
			self.AgentOutput = self.DoneAgentOutput

	@observe(ignore_input=True, ignore_output=False)
	async def _judge_trace(self) -> JudgementResult | None:
		"""Judge the trace of the agent"""
		task = self.task
		final_result = self.history.final_result() or ''
		agent_steps = self.history.agent_steps()
		screenshot_paths = [p for p in self.history.screenshot_paths() if p is not None]

		# Construct input messages for judge evaluation
		input_messages = construct_judge_messages(
			task=task,
			final_result=final_result,
			agent_steps=agent_steps,
			screenshot_paths=screenshot_paths,
			max_images=10,
			ground_truth=self.settings.ground_truth,
			use_vision=self.settings.use_vision,
		)

		# Call LLM with JudgementResult as output format
		kwargs: dict = {'output_format': JudgementResult}

		# Only pass request_type for ChatBrowserUse (other providers don't support it)
		if self.judge_llm.provider == 'browser-use':
			kwargs['request_type'] = 'judge'
			kwargs['session_id'] = self.session_id

		try:
			response = await self.judge_llm.ainvoke(input_messages, **kwargs)
			judgement: JudgementResult = response.completion  # type: ignore[assignment]
			return judgement
		except Exception as e:
			self.logger.error(f'Judge trace failed: {e}')
			# Return a default judgement on failure
			return None

	async def _judge_and_log(self) -> None:
		"""Run judge evaluation and log the verdict.

		The judge verdict is attached to the action result but does NOT override
		last_result.success — that stays as the agent's self-report. Telemetry
		sends both values so the eval platform can compare agent vs judge.
		"""
		judgement = await self._judge_trace()

		# Attach judgement to last action result
		if self.history.history[-1].result[-1].is_done:
			last_result = self.history.history[-1].result[-1]
			last_result.judgement = judgement

			# Get self-reported success
			self_reported_success = last_result.success

			# Log the verdict based on self-reported success and judge verdict
			if judgement:
				# If both self-reported and judge agree on success, don't log
				if self_reported_success is True and judgement.verdict is True:
					return

				judge_log = '\n'
				# If agent reported success but judge thinks it failed, show warning
				if self_reported_success is True and judgement.verdict is False:
					judge_log += '⚠️  \033[33mAgent reported success but judge thinks task failed\033[0m\n'

				# Otherwise, show full judge result
				verdict_color = '\033[32m' if judgement.verdict else '\033[31m'
				verdict_text = '✅ PASS' if judgement.verdict else '❌ FAIL'
				judge_log += f'⚖️  {verdict_color}Judge Verdict: {verdict_text}\033[0m\n'
				if judgement.failure_reason:
					judge_log += f'   Failure Reason: {judgement.failure_reason}\n'
				if judgement.reached_captcha:
					self.logger.warning(
						'Agent was blocked by a captcha. Cloud browsers include stealth fingerprinting and proxy rotation to avoid this.\n'
						'         Try: Browser(use_cloud=True)  |  Get an API key: https://cloud.browser-use.com?utm_source=oss&utm_medium=captcha_nudge'
					)
				judge_log += f'   {judgement.reasoning}\n'
				self.logger.info(judge_log)

	async def _get_model_output_with_retry(self, input_messages: list[BaseMessage]) -> AgentOutput:
		"""Get model output with retry logic for empty actions"""
		model_output = await self.get_model_output(input_messages)
		self.logger.debug(
			f'✅ Step {self.state.n_steps}: Got LLM response with {len(model_output.action) if model_output.action else 0} actions'
		)

		if (
			not model_output.action
			or not isinstance(model_output.action, list)
			or all(action.model_dump() == {} for action in model_output.action)
		):
			self.logger.warning('Model returned empty action. Retrying...')

			clarification_message = UserMessage(
				content='You forgot to return an action. Please respond with a valid JSON action according to the expected schema with your assessment and next actions.'
			)

			retry_messages = input_messages + [clarification_message]
			model_output = await self.get_model_output(retry_messages)

			if not model_output.action or all(action.model_dump() == {} for action in model_output.action):
				self.logger.warning('Model still returned empty after retry. Inserting safe noop action.')
				action_instance = self.ActionModel()
				setattr(
					action_instance,
					self._completion_action_name,
					{
						'actual_result': 'The executor returned no action, so the current UI state was not established.',
						'evidence': [],
						'action_completed': False,
						'side_effect_uncertain': False,
					},
				)
				model_output.action = [action_instance]

		return model_output

	async def _handle_post_llm_processing(
		self,
		browser_state_summary: BrowserStateSummary,
		input_messages: list[BaseMessage],
	) -> None:
		"""Handle callbacks and conversation saving after LLM interaction"""
		if self.register_new_step_callback and self.state.last_model_output:
			if inspect.iscoroutinefunction(self.register_new_step_callback):
				await self.register_new_step_callback(
					browser_state_summary,
					self.state.last_model_output,
					self.state.n_steps,
				)
			else:
				self.register_new_step_callback(
					browser_state_summary,
					self.state.last_model_output,
					self.state.n_steps,
				)

		if self.settings.save_conversation_path and self.state.last_model_output:
			# Treat save_conversation_path as a directory (consistent with other recording paths)
			conversation_dir = Path(self.settings.save_conversation_path)
			conversation_filename = f'conversation_{self.id}_{self.state.n_steps}.txt'
			target = conversation_dir / conversation_filename
			await save_conversation(
				input_messages,
				self.state.last_model_output,
				target,
				self.settings.save_conversation_path_encoding,
			)

	async def _make_history_item(
		self,
		model_output: AgentOutput | None,
		browser_state_summary: BrowserStateSummary,
		result: list[ActionResult],
		metadata: StepMetadata | None = None,
		state_message: str | None = None,
	) -> None:
		"""Create and store history item"""

		if model_output:
			interacted_elements = AgentHistory.get_interacted_element(model_output, browser_state_summary.dom_state.selector_map)
		else:
			interacted_elements = [None]

		# Store screenshot and get path
		screenshot_path = None
		if browser_state_summary.screenshot:
			self.logger.debug(
				f'📸 Storing screenshot for step {self.state.n_steps}, screenshot length: {len(browser_state_summary.screenshot)}'
			)
			screenshot_path = await self.screenshot_service.store_screenshot(browser_state_summary.screenshot, self.state.n_steps)
			self.logger.debug(f'📸 Screenshot stored at: {screenshot_path}')
		else:
			self.logger.debug(f'📸 No screenshot in browser_state_summary for step {self.state.n_steps}')

		state_history = BrowserStateHistory(
			url=browser_state_summary.url,
			title=browser_state_summary.title,
			tabs=browser_state_summary.tabs,
			interacted_element=interacted_elements,
			screenshot_path=screenshot_path,
		)

		history_item = AgentHistory(
			model_output=model_output,
			result=result,
			state=state_history,
			metadata=metadata,
			state_message=state_message,
		)

		self.history.add_item(history_item)

	def _remove_think_tags(self, text: str) -> str:
		THINK_TAGS = re.compile(r'<think>.*?</think>', re.DOTALL)
		STRAY_CLOSE_TAG = re.compile(r'.*?</think>', re.DOTALL)
		# Step 1: Remove well-formed <think>...</think>
		text = re.sub(THINK_TAGS, '', text)
		# Step 2: If there's an unmatched closing tag </think>,
		#         remove everything up to and including that.
		text = re.sub(STRAY_CLOSE_TAG, '', text)
		return text.strip()

	# region - URL replacement
	def _replace_urls_in_text(self, text: str) -> tuple[str, dict[str, str]]:
		"""Replace URLs in a text string"""

		replaced_urls: dict[str, str] = {}

		def replace_url(match: re.Match) -> str:
			"""Url can only have 1 query and 1 fragment"""
			import hashlib

			original_url = match.group(0)

			# Find where the query/fragment starts
			query_start = original_url.find('?')
			fragment_start = original_url.find('#')

			# Find the earliest position of query or fragment
			after_path_start = len(original_url)  # Default: no query/fragment
			if query_start != -1:
				after_path_start = min(after_path_start, query_start)
			if fragment_start != -1:
				after_path_start = min(after_path_start, fragment_start)

			# Split URL into base (up to path) and after_path (query + fragment)
			base_url = original_url[:after_path_start]
			after_path = original_url[after_path_start:]

			# If after_path is within the limit, don't shorten
			if len(after_path) <= self._url_shortening_limit:
				return original_url

			# If after_path is too long, truncate and add hash
			if after_path:
				truncated_after_path = after_path[: self._url_shortening_limit]
				# Create a short hash of the full after_path content
				hash_obj = hashlib.md5(after_path.encode('utf-8'))
				short_hash = hash_obj.hexdigest()[:7]
				# Create shortened URL
				shortened = f'{base_url}{truncated_after_path}...{short_hash}'
				# Only use shortened URL if it's actually shorter than the original
				if len(shortened) < len(original_url):
					replaced_urls[shortened] = original_url
					return shortened

			return original_url

		return URL_PATTERN.sub(replace_url, text), replaced_urls

	def _process_messsages_and_replace_long_urls_shorter_ones(self, input_messages: list[BaseMessage]) -> dict[str, str]:
		"""Replace long URLs with shorter ones
		? @dev edits input_messages in place

		returns:
			tuple[filtered_input_messages, urls we replaced {shorter_url: original_url}]
		"""
		from browser_use.llm.messages import AssistantMessage, UserMessage

		urls_replaced: dict[str, str] = {}

		# Process each message, in place
		for message in input_messages:
			# no need to process SystemMessage, we have control over that anyway
			if isinstance(message, (UserMessage, AssistantMessage)):
				if isinstance(message.content, str):
					# Simple string content
					message.content, replaced_urls = self._replace_urls_in_text(message.content)
					urls_replaced.update(replaced_urls)

				elif isinstance(message.content, list):
					# List of content parts
					for part in message.content:
						if isinstance(part, ContentPartTextParam):
							part.text, replaced_urls = self._replace_urls_in_text(part.text)
							urls_replaced.update(replaced_urls)

		return urls_replaced

	@staticmethod
	def _recursive_process_all_strings_inside_pydantic_model(model: BaseModel, url_replacements: dict[str, str]) -> None:
		"""Recursively process all strings inside a Pydantic model, replacing shortened URLs with originals in place."""
		for field_name, field_value in model.__dict__.items():
			if isinstance(field_value, str):
				# Replace shortened URLs with original URLs in string
				processed_string = Agent._replace_shortened_urls_in_string(field_value, url_replacements)
				setattr(model, field_name, processed_string)
			elif isinstance(field_value, BaseModel):
				# Recursively process nested Pydantic models
				Agent._recursive_process_all_strings_inside_pydantic_model(field_value, url_replacements)
			elif isinstance(field_value, dict):
				# Process dictionary values in place
				Agent._recursive_process_dict(field_value, url_replacements)
			elif isinstance(field_value, (list, tuple)):
				processed_value = Agent._recursive_process_list_or_tuple(field_value, url_replacements)
				setattr(model, field_name, processed_value)

	@staticmethod
	def _recursive_process_dict(dictionary: dict, url_replacements: dict[str, str]) -> None:
		"""Helper method to process dictionaries."""
		for k, v in dictionary.items():
			if isinstance(v, str):
				dictionary[k] = Agent._replace_shortened_urls_in_string(v, url_replacements)
			elif isinstance(v, BaseModel):
				Agent._recursive_process_all_strings_inside_pydantic_model(v, url_replacements)
			elif isinstance(v, dict):
				Agent._recursive_process_dict(v, url_replacements)
			elif isinstance(v, (list, tuple)):
				dictionary[k] = Agent._recursive_process_list_or_tuple(v, url_replacements)

	@staticmethod
	def _recursive_process_list_or_tuple(container: list | tuple, url_replacements: dict[str, str]) -> list | tuple:
		"""Helper method to process lists and tuples."""
		if isinstance(container, tuple):
			# For tuples, create a new tuple with processed items
			processed_items = []
			for item in container:
				if isinstance(item, str):
					processed_items.append(Agent._replace_shortened_urls_in_string(item, url_replacements))
				elif isinstance(item, BaseModel):
					Agent._recursive_process_all_strings_inside_pydantic_model(item, url_replacements)
					processed_items.append(item)
				elif isinstance(item, dict):
					Agent._recursive_process_dict(item, url_replacements)
					processed_items.append(item)
				elif isinstance(item, (list, tuple)):
					processed_items.append(Agent._recursive_process_list_or_tuple(item, url_replacements))
				else:
					processed_items.append(item)
			return tuple(processed_items)
		else:
			# For lists, modify in place
			for i, item in enumerate(container):
				if isinstance(item, str):
					container[i] = Agent._replace_shortened_urls_in_string(item, url_replacements)
				elif isinstance(item, BaseModel):
					Agent._recursive_process_all_strings_inside_pydantic_model(item, url_replacements)
				elif isinstance(item, dict):
					Agent._recursive_process_dict(item, url_replacements)
				elif isinstance(item, (list, tuple)):
					container[i] = Agent._recursive_process_list_or_tuple(item, url_replacements)
			return container

	@staticmethod
	def _replace_shortened_urls_in_string(text: str, url_replacements: dict[str, str]) -> str:
		"""Replace all shortened URLs in a string with their original URLs."""
		result = text
		for shortened_url, original_url in url_replacements.items():
			result = result.replace(shortened_url, original_url)
		return result

	# endregion - URL replacement

	@time_execution_async('--get_next_action')
	@observe_debug(ignore_input=True, ignore_output=True, name='get_model_output')
	async def get_model_output(self, input_messages: list[BaseMessage]) -> AgentOutput:
		"""Get next action from LLM based on current state"""

		urls_replaced = self._process_messsages_and_replace_long_urls_shorter_ones(input_messages)

		# Build kwargs for ainvoke
		# Note: ChatBrowserUse will automatically generate action descriptions from output_format schema
		kwargs: dict = {'output_format': self.AgentOutput, 'session_id': self.session_id}

		try:
			self.logger.info(
				'QA executor LLM request (step=%s, model=%s):\n%s',
				self.state.n_steps,
				getattr(self.llm, 'model', 'unknown'),
				_messages_for_llm_log(input_messages),
			)
			response = await self.llm.ainvoke(input_messages, **kwargs)
			parsed: AgentOutput = response.completion  # type: ignore[assignment]

			# Replace any shortened URLs in the LLM response back to original URLs
			if urls_replaced:
				self._recursive_process_all_strings_inside_pydantic_model(parsed, urls_replaced)
			self.logger.info(
				'QA executor LLM response (step=%s):\n%s',
				self.state.n_steps,
				_model_output_for_llm_log(parsed),
			)

			# cut the number of actions to max_actions_per_step if needed
			if len(parsed.action) > self.settings.max_actions_per_step:
				parsed.action = parsed.action[: self.settings.max_actions_per_step]

			if not (hasattr(self.state, 'paused') and (self.state.paused or self.state.stopped)):
				log_response(parsed, self.tools.registry.registry, self.logger)
				await self._broadcast_model_state(parsed)

			self._log_next_action_summary(parsed)
			return parsed
		except ValidationError:
			# Just re-raise - Pydantic's validation errors are already descriptive
			raise
		except (ModelRateLimitError, ModelProviderError) as e:
			# Check if we can switch to a fallback LLM
			if not self._try_switch_to_fallback_llm(e):
				# No fallback available, re-raise the original error
				raise
			# Retry with the fallback LLM
			return await self.get_model_output(input_messages)

	def _try_switch_to_fallback_llm(self, error: ModelRateLimitError | ModelProviderError) -> bool:
		"""
		Attempt to switch to a fallback LLM after a rate limit or provider error.

		Returns True if successfully switched to a fallback, False if no fallback available.
		Once switched, the agent will use the fallback LLM for the rest of the run.
		"""
		# Already using fallback - can't switch again
		if self._using_fallback_llm:
			self.logger.warning(
				f'⚠️ Fallback LLM also failed ({type(error).__name__}: {error.message}), no more fallbacks available'
			)
			return False

		# Check if error is retryable (rate limit, auth errors, or server errors)
		# 401: API key invalid/expired - fallback to different provider
		# 402: Insufficient credits/payment required - fallback to different provider
		# 429: Rate limit exceeded
		# 500, 502, 503, 504: Server errors
		# ModelOutputTruncatedError: not retryable on the same model, but a fallback may have a higher cap
		retryable_status_codes = {401, 402, 429, 500, 502, 503, 504}
		is_retryable = isinstance(error, (ModelRateLimitError, ModelOutputTruncatedError)) or (
			hasattr(error, 'status_code') and error.status_code in retryable_status_codes
		)

		if not is_retryable:
			return False

		# Check if we have a fallback LLM configured
		if self._fallback_llm is None:
			self.logger.warning(f'⚠️ LLM error ({type(error).__name__}: {error.message}) but no fallback_llm configured')
			return False

		self._log_fallback_switch(error, self._fallback_llm)

		# Switch to the fallback LLM
		self.llm = self._fallback_llm
		self._using_fallback_llm = True

		# Register the fallback LLM for token cost tracking
		self.token_cost_service.register_llm(self._fallback_llm)

		return True

	def _log_fallback_switch(self, error: ModelRateLimitError | ModelProviderError, fallback: BaseChatModel) -> None:
		"""Log when switching to a fallback LLM."""
		original_model = self._original_llm.model if hasattr(self._original_llm, 'model') else 'unknown'
		fallback_model = fallback.model if hasattr(fallback, 'model') else 'unknown'
		error_type = type(error).__name__
		status_code = getattr(error, 'status_code', 'N/A')

		self.logger.warning(
			f'⚠️ Primary LLM ({original_model}) failed with {error_type} (status={status_code}), '
			f'switching to fallback LLM ({fallback_model})'
		)

	async def _log_agent_run(self) -> None:
		"""Log the agent run"""
		# Blue color for task
		self.logger.info(f'\033[34m🎯 Task: {self.task}\033[0m')

		self.logger.debug(f'🤖 Browser-Use Library Version {self.version} ({self.source})')

		# Check for latest version and log upgrade message if needed
		if CONFIG.BROWSER_USE_VERSION_CHECK:
			latest_version = await check_latest_browser_use_version()
			if latest_version and latest_version != self.version:
				self.logger.info(
					f'📦 Newer version available: {latest_version} (current: {self.version}). Upgrade with: uv add browser-use=={latest_version}'
				)

	def _log_first_step_startup(self) -> None:
		"""Log startup message only on the first step"""
		if len(self.history.history) == 0:
			self.logger.info(
				f'Starting a browser-use agent with version {self.version}, with provider={self.llm.provider} and model={self.llm.model}'
			)

	def _log_step_context(self, browser_state_summary: BrowserStateSummary) -> None:
		"""Log step context information"""
		url = browser_state_summary.url if browser_state_summary else ''
		url_short = url[:50] + '...' if len(url) > 50 else url
		interactive_count = len(browser_state_summary.dom_state.selector_map) if browser_state_summary else 0
		self.logger.info('\n')
		self.logger.info(f'📍 Step {self.state.n_steps}:')
		self.logger.debug(f'Evaluating page with {interactive_count} interactive elements on: {url_short}')

	def _log_next_action_summary(self, parsed: 'AgentOutput') -> None:
		"""Log a comprehensive summary of the next action(s)"""
		if not (self.logger.isEnabledFor(logging.DEBUG) and parsed.action):
			return

		# Collect action details
		action_details = []
		for i, action in enumerate(parsed.action):
			action_data = action.model_dump(exclude_unset=True)
			action_name = next(iter(action_data.keys())) if action_data else 'unknown'
			action_params = action_data.get(action_name, {}) if action_data else {}

			# Format key parameters concisely
			param_summary = []
			if isinstance(action_params, dict):
				for key, value in action_params.items():
					if key == 'index':
						param_summary.append(f'#{value}')
					elif key == 'text' and isinstance(value, str):
						text_preview = value[:30] + '...' if len(value) > 30 else value
						param_summary.append(f'text="{text_preview}"')
					elif key == 'url':
						param_summary.append(f'url="{value}"')
					elif key == 'success':
						param_summary.append(f'success={value}')
					elif isinstance(value, (str, int, bool)):
						val_str = str(value)[:30] + '...' if len(str(value)) > 30 else str(value)
						param_summary.append(f'{key}={val_str}')

			param_str = f'({", ".join(param_summary)})' if param_summary else ''
			action_details.append(f'{action_name}{param_str}')

	def _prepare_demo_message(self, message: str, limit: int = 600) -> str:
		# Previously truncated long entries; keep full text for better context in demo panel
		return message.strip()

	async def _demo_mode_log(self, message: str, level: str = 'info', metadata: dict[str, Any] | None = None) -> None:
		if not self._demo_mode_enabled or not message or self.browser_session is None:
			return
		try:
			await self.browser_session.send_demo_mode_log(
				message=self._prepare_demo_message(message),
				level=level,
				metadata=metadata or {},
			)
		except Exception as exc:
			self.logger.debug(f'[DemoMode] Failed to send overlay log: {exc}')

	async def _broadcast_model_state(self, parsed: 'AgentOutput') -> None:
		if not self._demo_mode_enabled:
			return

		state = parsed.current_state
		step_meta = {'step': self.state.n_steps}

		if state.thinking:
			await self._demo_mode_log(state.thinking, 'thought', step_meta)

		if state.evaluation_previous_goal:
			eval_text = state.evaluation_previous_goal
			level = 'success' if 'success' in eval_text.lower() else 'warning' if 'failure' in eval_text.lower() else 'info'
			await self._demo_mode_log(eval_text, level, step_meta)

		if state.memory:
			await self._demo_mode_log(f'Memory: {state.memory}', 'info', step_meta)

		if state.next_goal:
			await self._demo_mode_log(f'Next goal: {state.next_goal}', 'info', step_meta)

	def _log_step_completion_summary(self, step_start_time: float, result: list[ActionResult]) -> str | None:
		"""Log step completion summary with action count, timing, and success/failure stats"""
		if not result:
			return None

		step_duration = time.time() - step_start_time
		action_count = len(result)

		# Count success and failures
		success_count = sum(1 for r in result if not r.error)
		failure_count = action_count - success_count

		# Format success/failure indicators
		success_indicator = f'✅ {success_count}' if success_count > 0 else ''
		failure_indicator = f'❌ {failure_count}' if failure_count > 0 else ''
		status_parts = [part for part in [success_indicator, failure_indicator] if part]
		status_str = ' | '.join(status_parts) if status_parts else '✅ 0'

		message = (
			f'📍 Step {self.state.n_steps}: Ran {action_count} action{"" if action_count == 1 else "s"} '
			f'in {step_duration:.2f}s: {status_str}'
		)
		self.logger.debug(message)
		return message

	def _log_final_outcome_messages(self) -> None:
		"""Log helpful messages to user based on agent run outcome"""
		qa_result = self.history.qa_result
		if qa_result is not None:
			if qa_result.status == QARunStatus.PASSED:
				self.logger.info(f'✅ {qa_result.summary}')
				return
			if qa_result.status == QARunStatus.SUT_FAILED:
				self.logger.info(f'❌ SUT expectation failed: {qa_result.summary}')
				return
			self.logger.warning(f'⚠️ QA run ended without a product verdict: {qa_result.summary}')
			return

		# Check if agent failed
		is_successful = self.history.is_successful()

		if is_successful is False or is_successful is None:
			# Get final result to check for specific failure reasons
			final_result = self.history.final_result()
			final_result_str = str(final_result).lower() if final_result else ''

			# Check for captcha/cloudflare related failures
			captcha_keywords = ['captcha', 'cloudflare', 'recaptcha', 'challenge', 'bot detection', 'access denied']
			has_captcha_issue = any(keyword in final_result_str for keyword in captcha_keywords)

			if has_captcha_issue:
				self.logger.warning(
					'Agent was blocked by a captcha. Cloud browsers include stealth fingerprinting and proxy rotation to avoid this.\n'
					'         Try: Browser(use_cloud=True)  |  Get an API key: https://cloud.browser-use.com?utm_source=oss&utm_medium=captcha_nudge'
				)

			# General failure message
			self.logger.info('')
			self.logger.info('Did the Agent not work as expected? Let us fix this!')
			self.logger.info('   Open a short issue on GitHub: https://github.com/browser-use/browser-use/issues')

	def _log_agent_event(self, max_steps: int, agent_run_error: str | None = None) -> None:
		"""Sent the agent event for this run to telemetry"""

		token_summary = self.token_cost_service.get_usage_tokens_for_model(self.llm.model)

		# Prepare action_history data correctly
		action_history_data = []
		for item in self.history.history:
			if item.model_output and item.model_output.action:
				# Convert each ActionModel in the step to its dictionary representation
				step_actions = [
					action.model_dump(exclude_unset=True)
					for action in item.model_output.action
					if action  # Ensure action is not None if list allows it
				]
				action_history_data.append(step_actions)
			else:
				# Append None or [] if a step had no actions or no model output
				action_history_data.append(None)

		final_res = self.history.final_result()
		final_result_str = json.dumps(final_res) if final_res is not None else None

		# Extract judgement data if available
		judgement_data = self.history.judgement()
		judge_verdict = judgement_data.get('verdict') if judgement_data else None
		judge_reasoning = judgement_data.get('reasoning') if judgement_data else None
		judge_failure_reason = judgement_data.get('failure_reason') if judgement_data else None
		judge_reached_captcha = judgement_data.get('reached_captcha') if judgement_data else None
		judge_impossible_task = judgement_data.get('impossible_task') if judgement_data else None
		qa_result = self.history.qa_result

		self.telemetry.capture(
			AgentTelemetryEvent(
				task=self.task,
				model=self.llm.model,
				model_provider=self.llm.provider,
				max_steps=max_steps,
				max_actions_per_step=self.settings.max_actions_per_step,
				use_vision=self.settings.use_vision,
				version=self.version,
				source=self.source,
				cdp_url=urlparse(self.browser_session.cdp_url).hostname
				if self.browser_session and self.browser_session.cdp_url
				else None,
				agent_type=None,  # Regular Agent (not code-use)
				action_errors=self.history.errors(),
				action_history=action_history_data,
				urls_visited=self.history.urls(),
				steps=self.state.n_steps,
				total_input_tokens=token_summary.prompt_tokens,
				total_output_tokens=token_summary.completion_tokens,
				prompt_cached_tokens=token_summary.prompt_cached_tokens,
				total_tokens=token_summary.total_tokens,
				total_duration_seconds=self.history.total_duration_seconds(),
				success=self.history.is_successful(),
				final_result_response=final_result_str,
				error_message=agent_run_error,
				judge_verdict=judge_verdict,
				judge_reasoning=judge_reasoning,
				judge_failure_reason=judge_failure_reason,
				judge_reached_captcha=judge_reached_captcha,
				judge_impossible_task=judge_impossible_task,
				qa_status=qa_result.status.value if qa_result else None,
				qa_failure_origin=qa_result.failure_origin.value if qa_result and qa_result.failure_origin else None,
				qa_stopped_at_step=qa_result.stopped_at_step if qa_result else None,
				qa_total_steps=len(qa_result.test_case.steps) if qa_result and qa_result.test_case else None,
				qa_executed_steps=(
					sum(1 for result in qa_result.step_results if result.status != QAStepStatus.NOT_RUN) if qa_result else None
				),
			)
		)

	async def take_step(self, step_info: AgentStepInfo | None = None) -> tuple[bool, bool]:
		"""Reject bypassing QA compilation and per-step independent judgement."""

		raise RuntimeError('browser_use.Agent.take_step() cannot bypass the QA runner; use Agent.run()')

	def _extract_start_url(self, task: str) -> str | None:
		"""Extract URL from task string using naive pattern matching."""

		import re

		# Remove email addresses from task before looking for URLs
		task_without_emails = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', task)

		# Look for common URL patterns
		patterns = [
			r'(?:https?|file)://[^\s<>"\']+',  # Full URLs
			r'(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}(?:/[^\s<>"\']*)?',  # Domain names with subdomains and optional paths
		]

		# File extensions that should be excluded from URL detection
		# These are likely files rather than web pages to navigate to
		excluded_extensions = {
			# Documents
			'pdf',
			'doc',
			'docx',
			'xls',
			'xlsx',
			'ppt',
			'pptx',
			'odt',
			'ods',
			'odp',
			# Text files
			'txt',
			'md',
			'csv',
			'json',
			'xml',
			'yaml',
			'yml',
			# Archives
			'zip',
			'rar',
			'7z',
			'tar',
			'gz',
			'bz2',
			'xz',
			# Images
			'jpg',
			'jpeg',
			'png',
			'gif',
			'bmp',
			'svg',
			'webp',
			'ico',
			# Audio/Video
			'mp3',
			'mp4',
			'avi',
			'mkv',
			'mov',
			'wav',
			'flac',
			'ogg',
			# Code/Data
			'py',
			'js',
			'css',
			'java',
			'cpp',
			# Academic/Research
			'bib',
			'bibtex',
			'tex',
			'latex',
			'cls',
			'sty',
			# Other common file types
			'exe',
			'msi',
			'dmg',
			'pkg',
			'deb',
			'rpm',
			'iso',
			# GitHub/Project paths
			'polynomial',
		}

		excluded_words = {
			'never',
			'dont',
			'not',
			"don't",
		}

		found_urls = []
		matched_spans: list[tuple[int, int]] = []
		for pattern in patterns:
			matches = re.finditer(pattern, task_without_emails)
			for match in matches:
				url = match.group(0)
				original_position = match.start()  # Store original position before URL modification

				# Skip fragments of URLs already matched by earlier pattern
				if any(match.start() < end and match.end() > start for start, end in matched_spans):
					continue
				matched_spans.append((match.start(), match.end()))

				# Remove trailing punctuation that's not part of URLs
				url = sanitize_url_candidate(url)

				if is_placeholder_url(url):
					self.logger.debug(f'Excluding placeholder URL from auto-navigation: {url}')
					continue

				url_lower = url.lower()
				has_scheme = url_lower.startswith(('http://', 'https://', 'file://'))

				# Check if URL ends with file extension
				should_exclude = False
				if not url_lower.startswith('file://'):
					for ext in excluded_extensions:
						if f'.{ext}' in url_lower:
							should_exclude = True
							break
					if not has_scheme and '.htm' in url_lower:
						should_exclude = True

				if should_exclude:
					self.logger.debug(f'Excluding URL with file extension from auto-navigation: {url}')
					continue

				# If in the 20 characters before the url position is a word in excluded_words skip to avoid "Never go to this url"
				context_start = max(0, original_position - 20)
				context_text = task_without_emails[context_start:original_position]
				if any(word.lower() in context_text.lower() for word in excluded_words):
					self.logger.debug(
						f'Excluding URL with word in excluded words from auto-navigation: {url} (context: "{context_text.strip()}")'
					)
					continue

				# Add https:// if missing (after excluded words check to avoid position calculation issues)
				if not has_scheme:
					url = 'https://' + url

				found_urls.append(url)

		unique_urls = list(set(found_urls))
		# If multiple URLs found, skip directly_open_urling
		if len(unique_urls) > 1:
			self.logger.debug(f'Multiple URLs found ({len(found_urls)}), skipping directly_open_url to avoid ambiguity')
			return None

		# If exactly one URL found, return it
		if len(unique_urls) == 1:
			return unique_urls[0]

		return None

	async def _execute_step(
		self,
		step: int,
		max_steps: int,
		step_info: AgentStepInfo,
		on_step_start: AgentHookFunc | None = None,
		on_step_end: AgentHookFunc | None = None,
	) -> bool:
		"""
		Execute a single step with timeout.

		Returns:
			bool: True if task is done, False otherwise
		"""
		if on_step_start is not None:
			await on_step_start(self)

		await self._demo_mode_log(
			f'Starting step {step + 1}/{max_steps}',
			'info',
			{'step': step + 1, 'total_steps': max_steps},
		)

		self.logger.debug(f'🚶 Starting step {step + 1}/{max_steps}...')

		try:
			await asyncio.wait_for(
				self.step(step_info),
				timeout=self.settings.step_timeout,
			)
			self.logger.debug(f'✅ Completed step {step + 1}/{max_steps}')
		except TimeoutError:
			# Handle step timeout gracefully
			error_msg = f'Step {step + 1} timed out after {self.settings.step_timeout} seconds'
			self.logger.error(f'⏰ {error_msg}')
			await self._demo_mode_log(error_msg, 'error', {'step': step + 1})
			self.state.consecutive_failures += 1
			self.state.last_result = [ActionResult(error=error_msg)]
			# Ensure step counter advances on timeout — _finalize() may have
			# been skipped or returned early due to the cancellation.
			if self.state.n_steps == step + 1:
				self.state.n_steps += 1

		if on_step_end is not None:
			await on_step_end(self)

		if self.history.is_done():
			# In QA mode this is only a business-step boundary. run() invokes the
			# independent judge and decides whether to advance, retry, or terminate.
			return True

		return False

	def _qa_step_prompt(self, step: WebUITestStep, *, retry_reason: str | None = None) -> str:
		assert self._qa_test_case is not None
		case_preconditions = [str(getattr(item, 'description', item)) for item in self._qa_test_case.preconditions]
		step_preconditions = [str(getattr(item, 'description', item)) for item in step.preconditions]
		retry_section = ''
		if retry_reason:
			retry_section = f'\nPrevious attempt was not objectively completed. Recovery reason: {retry_reason}\n'
		phase_heading = (
			f'QA CLEANUP {self._qa_cleanup_index + 1}/{len(self._qa_test_case.cleanup_steps)}'
			if self._qa_running_cleanup
			else f'QA STEP {self._qa_current_step_index + 1}/{len(self._qa_test_case.steps)}'
		)
		resolved_test_data = {
			key: value.replace('${run_id}', self._qa_run_id) for key, value in self._qa_test_case.test_data.items()
		}
		idempotency_key = step.idempotency_key.replace('${run_id}', self._qa_run_id) if step.idempotency_key else None
		return f"""{phase_heading}
Step ID: {step.step_id}
Run ID: {self._qa_run_id}
Idempotency key: {idempotency_key}
Non-sensitive test data: {resolved_test_data}
Test case preconditions: {case_preconditions}
Business action: {step.instruction}
Expected observable result: {step.expected_result}
Expectation source: {step.expectation_source.value}
Step preconditions: {step_preconditions}
{retry_section}
Execute only this business step. Use the current page as the source of truth. When the resulting state is observable,
call finish_test_step alone with objective evidence. Do not decide pass/fail and do not continue to another business step."""

	def _qa_preconditions_prompt(self, step: WebUITestStep) -> str:
		assert self._qa_test_case is not None
		lines = [
			f'- {item.mode.value.upper()} {item.precondition_id}: {item.description}' for item in self._qa_test_case.preconditions
		]
		return f"""QA PRECONDITION PHASE
Required conditions:
{chr(10).join(lines)}

VERIFY conditions must only be inspected. ENSURE conditions may perform the stated setup, such as authentication.
Do not execute business test steps. If a required credential, role, test datum, CAPTCHA, or dependency blocks setup,
report that objective state. When all conditions are satisfied or a blocker is observable, call finish_test_step alone.
Expected state: {step.expected_result}"""

	def _available_sensitive_reference_names(self) -> set[str]:
		names: set[str] = set()
		for key, value in (self.sensitive_data or {}).items():
			if isinstance(value, dict):
				names.update(str(nested_key) for nested_key in value)
			else:
				names.add(str(key))
		return names

	def _redact_qa_text(self, value: str) -> str:
		"""Redact registered sensitive values before evidence reaches disk or a Judge."""

		sensitive_values = collect_sensitive_data_values(self.sensitive_data)
		return redact_sensitive_string(value, sensitive_values) if sensitive_values else value

	def _create_qa_artifact(
		self,
		*,
		kind: EvidenceKind,
		summary: str,
		url: str | None = None,
		artifact_path: str | None = None,
		metadata: dict[str, Any] | None = None,
		redacted: bool = True,
	) -> EvidenceArtifact:
		"""Create a content-addressed evidence record for one captured observation."""

		redacted_summary = self._redact_qa_text(summary)
		digest_source = redacted_summary.encode('utf-8')
		if artifact_path:
			try:
				digest_source = Path(artifact_path).read_bytes()
			except OSError:
				pass
		return EvidenceArtifact(
			kind=kind,
			tab_id=self.browser_session.agent_focus_target_id,
			url=url,
			summary=redacted_summary[:4000],
			artifact_path=artifact_path,
			sha256=hashlib.sha256(digest_source).hexdigest(),
			redacted=redacted,
			metadata=metadata or {},
		)

	async def _capture_qa_snapshot(self, label: str, *, include_screenshot: bool | None = None) -> BrowserEvidenceSnapshot:
		if include_screenshot is None:
			include_screenshot = self.settings.use_vision is not False
		state = await self.browser_session.get_browser_state_summary(
			include_screenshot=include_screenshot, include_recent_events=True
		)
		diagnostics = self._qa_evidence_monitor.since(self._qa_evidence_cursor) if self._qa_evidence_monitor is not None else None
		dom_summary = ''
		try:
			dom_summary = self._redact_qa_text(
				state.dom_state.llm_representation(include_attributes=self.settings.include_attributes)[:30000]
			)
		except Exception as exc:
			dom_summary = f'Unable to serialize DOM evidence: {type(exc).__name__}: {exc}'

		evidence_dir = self.agent_directory / 'qa_evidence'
		evidence_dir.mkdir(parents=True, exist_ok=True)
		safe_label = re.sub(r'[^A-Za-z0-9_.-]+', '_', label)
		artifacts: list[EvidenceArtifact] = []
		dom_path = evidence_dir / f'{safe_label}.dom.txt'
		try:
			dom_path.write_text(dom_summary, encoding='utf-8')
			artifacts.append(
				self._create_qa_artifact(
					kind=EvidenceKind.DOM,
					summary=dom_summary,
					url=state.url,
					artifact_path=str(dom_path),
					metadata={'label': label},
				)
			)
		except OSError as exc:
			self.logger.warning(f'Failed to persist QA DOM evidence: {exc}')
		artifacts.append(self._create_qa_artifact(kind=EvidenceKind.URL, summary=state.url, url=state.url))

		screenshot_path: str | None = None
		# Screenshots can contain secrets that cannot be reliably text-redacted. When
		# sensitive values are registered, rely on redacted DOM evidence instead.
		if state.screenshot and not collect_sensitive_data_values(self.sensitive_data):
			try:
				path = evidence_dir / f'{safe_label}.png'
				path.write_bytes(base64.b64decode(state.screenshot))
				screenshot_path = str(path)
				artifacts.append(
					self._create_qa_artifact(
						kind=EvidenceKind.SCREENSHOT,
						summary=f'Screenshot captured at {state.url}',
						url=state.url,
						artifact_path=screenshot_path,
						redacted=True,
					)
				)
			except Exception as exc:
				self.logger.warning(f'Failed to persist QA screenshot evidence: {exc}')

		browser_errors = [self._redact_qa_text(str(item)) for item in state.browser_errors]
		network_errors = [self._redact_qa_text(str(item)) for item in diagnostics.network_errors] if diagnostics else []
		console_errors = [self._redact_qa_text(str(item)) for item in diagnostics.console_errors] if diagnostics else []
		for kind, messages in (
			(EvidenceKind.BROWSER, browser_errors),
			(EvidenceKind.NETWORK, network_errors),
			(EvidenceKind.CONSOLE, console_errors),
		):
			for message in messages:
				artifacts.append(self._create_qa_artifact(kind=kind, summary=message, url=state.url))
		if diagnostics:
			for event in diagnostics.network_events:
				safe_url = self._redact_qa_text(event.url)
				summary = (
					f'{event.method} {event.status if event.status is not None else "FAILED"} [{event.resource_type}] {safe_url}'
				)
				artifacts.append(
					self._create_qa_artifact(
						kind=EvidenceKind.NETWORK,
						summary=summary,
						url=state.url,
						metadata={
							'request_id': event.request_id,
							'method': event.method,
							'url': safe_url,
							'resource_type': event.resource_type,
							'status': event.status,
							'error': self._redact_qa_text(event.error or '') or None,
						},
					)
				)

		return BrowserEvidenceSnapshot(
			url=state.url,
			title=state.title,
			dom_summary=dom_summary,
			screenshot_path=screenshot_path,
			browser_errors=browser_errors,
			network_errors=network_errors,
			console_errors=console_errors,
			recent_events=self._redact_qa_text(state.recent_events) if state.recent_events else None,
			artifacts=artifacts,
		)

	async def _capture_qa_lightweight_snapshot(self) -> BrowserEvidenceSnapshot:
		"""Capture a replay boundary without rebuilding the full DOM tree.

		The deterministic replay judge only evaluates post-action assertions. The
		pre-action snapshot therefore needs the cached page identity and diagnostic
		cursor, while the full DOM remains reserved for post-action evidence.
		"""

		diagnostics = self._qa_evidence_monitor.since(self._qa_evidence_cursor) if self._qa_evidence_monitor else None
		return BrowserEvidenceSnapshot(
			url=await self.browser_session.get_current_page_url(),
			title=await self.browser_session.get_current_page_title(),
			network_errors=diagnostics.network_errors if diagnostics else [],
			console_errors=diagnostics.console_errors if diagnostics else [],
		)

	async def _wait_for_qa_page_stability(self, *, timeout: float = 8.0, stable_window: float = 0.4) -> None:
		"""Wait for URL/title/DOM state to stabilize instead of using a fixed sleep."""

		deadline = time.monotonic() + timeout
		last_signature: str | None = None
		stable_since = time.monotonic()
		while time.monotonic() < deadline:
			state = await self.browser_session.get_browser_state_summary(include_screenshot=False)
			try:
				dom = state.dom_state.llm_representation(include_attributes=self.settings.include_attributes)[:12000]
			except Exception:
				dom = ''
			signature = hashlib.sha256(f'{state.url}\0{state.title}\0{dom}'.encode()).hexdigest()
			now = time.monotonic()
			if signature != last_signature:
				last_signature = signature
				stable_since = now
			elif now - stable_since >= stable_window:
				return
			await asyncio.sleep(0.1)
		self._qa_warnings.append(f'Page did not reach a stable DOM window within {timeout:.1f}s')

	@staticmethod
	def _qa_storage_maps(storage_state: dict[str, Any], root_origin: str) -> tuple[dict[str, str], dict[str, str]]:
		for origin in storage_state.get('origins', []):
			if origin.get('origin') != root_origin:
				continue
			local = {str(item['name']): str(item['value']) for item in origin.get('localStorage', [])}
			session = {str(item['name']): str(item['value']) for item in origin.get('sessionStorage', [])}
			return local, session
		return {}, {}

	def _filter_qa_login_storage_state(self, storage_state: dict[str, Any]) -> dict[str, Any]:
		"""Keep only cookies and web storage belonging to the current QA navigation scope."""

		if self._qa_scope is None:
			return {'cookies': [], 'origins': []}
		scope = self._qa_scope

		def cookie_domain_allowed(cookie: dict[str, Any]) -> bool:
			host = str(cookie.get('domain', '')).lstrip('.').rstrip('.').lower()
			if not host:
				return False
			if scope.exact_host_only:
				return host == scope.root_host
			return host == scope.registrable_domain or host.endswith(f'.{scope.registrable_domain}')

		cookies = [dict(cookie) for cookie in storage_state.get('cookies', []) if cookie_domain_allowed(dict(cookie))]
		origins = [dict(origin) for origin in storage_state.get('origins', []) if scope.allows(str(origin.get('origin', '')))]
		return {'cookies': cookies, 'origins': origins}

	async def _cache_qa_login_state(self) -> None:
		"""Capture in-scope authentication state in memory for a later same-Agent run."""

		if not self.settings.reuse_login_state or self._qa_scope is None or self.browser_session._cdp_client_root is None:
			return
		try:
			storage_state = await self.browser_session._cdp_get_storage_state()
			self._qa_login_storage_state = self._filter_qa_login_storage_state(storage_state)
		except Exception as exc:
			self.logger.debug(f'Could not cache QA login state: {type(exc).__name__}: {exc}')

	async def _restore_cached_qa_login_state(self) -> None:
		"""Restore cached in-scope credentials before the repeated run navigates to its root page."""

		if not self.settings.reuse_login_state or not self._qa_login_storage_state:
			return
		if not self._qa_login_storage_state.get('cookies') and not self._qa_login_storage_state.get('origins'):
			return
		try:
			cookies = []
			for raw_cookie in self._qa_login_storage_state.get('cookies', []):
				cookie = dict(raw_cookie)
				if cookie.get('expires') in {0, 0.0, -1, -1.0}:
					cookie.pop('expires', None)
				cookies.append(cookie)
			if cookies:
				await self.browser_session._cdp_set_cookies(cookies)

			for origin in self._qa_login_storage_state.get('origins', []):
				origin_value = origin.get('origin')
				if not origin_value:
					continue
				local_storage = {str(item['name']): str(item['value']) for item in origin.get('localStorage', [])}
				session_storage = {str(item['name']): str(item['value']) for item in origin.get('sessionStorage', [])}
				restore_script = f"""(() => {{
					if (window.location.origin !== {json.dumps(origin_value)}) return;
					const localValues = {json.dumps(local_storage)};
					const sessionValues = {json.dumps(session_storage)};
					for (const [key, value] of Object.entries(localValues)) window.localStorage.setItem(key, value);
					for (const [key, value] of Object.entries(sessionValues)) window.sessionStorage.setItem(key, value);
				}})();"""
				await self.browser_session._cdp_add_init_script(restore_script)
			self.logger.info('🔐 Reused in-scope login state from the previous QA run.')
		except Exception as exc:
			self.logger.warning(f'Could not restore cached QA login state; login may be required again: {exc}')

	async def _restore_qa_exploration_state(
		self,
		*,
		storage_state: dict[str, Any],
		original_target_id: str,
		original_tab_ids: set[str],
		root_origin: str,
	) -> None:
		"""Close exploration pages and restore cookies plus root-origin web storage."""

		try:
			current_tabs = await self.browser_session.get_tabs()
			if original_target_id in {tab.target_id for tab in current_tabs}:
				await self.browser_session.event_bus.dispatch(SwitchTabEvent(target_id=original_target_id))
			for tab in current_tabs:
				if tab.target_id not in original_tab_ids:
					await self.browser_session._cdp_close_page(tab.target_id)

			await self.browser_session._cdp_clear_cookies()
			cookies = storage_state.get('cookies', [])
			if cookies:
				await self.browser_session._cdp_set_cookies(cookies)

			await self.browser_session.navigate_to(self._qa_scope.root_url, new_tab=False)  # type: ignore[union-attr]
			await self._wait_for_qa_page_stability()
			local_storage, session_storage = self._qa_storage_maps(storage_state, root_origin)
			restore_script = f"""(() => {{
				if (window.location.origin !== {json.dumps(root_origin)}) return 'wrong-origin';
				localStorage.clear();
				sessionStorage.clear();
				const localItems = {json.dumps(local_storage)};
				const sessionItems = {json.dumps(session_storage)};
				for (const [key, value] of Object.entries(localItems)) localStorage.setItem(key, value);
				for (const [key, value] of Object.entries(sessionItems)) sessionStorage.setItem(key, value);
				return 'restored';
			}})()"""
			cdp_session = await self.browser_session.get_or_create_cdp_session()
			result = await cdp_session.cdp_client.send.Runtime.evaluate(
				params={'expression': restore_script, 'returnByValue': True}, session_id=cdp_session.session_id
			)
			if result.get('result', {}).get('value') != 'restored':
				raise QAExplorationStateError('Exploration storage restoration ran on an unexpected origin')

			# Reload so the SUT consumes the restored storage before formal execution.
			await self.browser_session.navigate_to(self._qa_scope.root_url, new_tab=False)  # type: ignore[union-attr]
			await self._wait_for_qa_page_stability()
			state = await self.browser_session.get_browser_state_summary(include_screenshot=False)
			if not self._qa_scope or not self._qa_scope.allows(state.url):
				raise QAExplorationStateError(f'Restored page left QA scope: {state.url}')
			verified_storage = await self.browser_session._cdp_get_storage_state()
			verified_local, verified_session = self._qa_storage_maps(verified_storage, root_origin)
			if verified_local != local_storage or verified_session != session_storage:
				raise QAExplorationStateError(
					'Page storage changed after restoration reload; clean formal state cannot be confirmed',
					inconclusive=True,
				)
		except QAExplorationStateError:
			raise
		except Exception as exc:
			raise QAExplorationStateError(f'Failed to restore read-only exploration state: {exc}') from exc

	def _qa_root_origin(self) -> str:
		"""Return the normalized origin used for root-scoped storage migration."""

		assert self._qa_scope is not None
		parsed_root = urlparse(self._qa_scope.root_url)
		root_host = self._qa_scope.root_host
		if ':' in root_host and not root_host.startswith('['):
			root_host = f'[{root_host}]'
		default_port = (parsed_root.scheme == 'https' and parsed_root.port in {None, 443}) or (
			parsed_root.scheme == 'http' and parsed_root.port in {None, 80}
		)
		return f'{parsed_root.scheme}://{root_host}' + ('' if default_port else f':{parsed_root.port}')

	async def _compile_qa_case_from_exploration_page(
		self,
		*,
		compiler: QATaskCompiler,
		draft: WebUITestCaseDraft,
	) -> WebUITestCase:
		"""Collect passive page evidence from the focused exploration target."""

		assert self._qa_scope is not None
		await self._wait_for_qa_page_stability()
		initial = await self.browser_session.get_browser_state_summary(include_screenshot=True, include_recent_events=True)
		if initial.state_error:
			raise RuntimeError(f'Unable to inspect QA root page: {initial.state_error}')
		await self.browser_session.event_bus.dispatch(ScrollEvent(direction='down', amount=700, node=None))
		await self._wait_for_qa_page_stability(timeout=2.0, stable_window=0.2)
		scrolled = await self.browser_session.get_browser_state_summary(include_screenshot=False, include_recent_events=True)
		await self.browser_session.event_bus.dispatch(ScrollEvent(direction='up', amount=700, node=None))
		dom_parts: list[str] = []
		for state in (initial, scrolled):
			try:
				dom_parts.append(state.dom_state.llm_representation(include_attributes=self.settings.include_attributes))
			except Exception as exc:
				raise RuntimeError(f'Unable to serialize QA discovery state: {exc}') from exc
		return await compiler.complete_with_discovery(
			draft=draft,
			scope=self._qa_scope,
			discovered_url=scrolled.url,
			discovered_title=scrolled.title,
			discovered_dom='\n\n--- SCROLLED READ-ONLY VIEW ---\n'.join(dom_parts),
		)

	async def _explore_in_isolated_browser_context(
		self,
		*,
		compiler: QATaskCompiler,
		draft: WebUITestCaseDraft,
		storage_state: dict[str, Any],
		original_target_id: str,
		root_origin: str,
	) -> WebUITestCase:
		"""Explore in a disposable CDP context containing only in-scope auth state."""

		assert self._qa_scope is not None
		root_client = self.browser_session._cdp_client_root
		if root_client is None:
			raise QAIsolatedContextUnavailable('Root CDP client is unavailable')

		context_id: str | None = None
		isolated_target_id: str | None = None
		context_ready = False
		try:
			created = await asyncio.wait_for(root_client.send.Target.createBrowserContext(params={}), timeout=8.0)
			context_id = created.get('browserContextId')
			if not context_id:
				raise QAIsolatedContextUnavailable('CDP did not return a browserContextId')

			filtered_storage = self._filter_qa_login_storage_state(storage_state)
			cookies = []
			for raw_cookie in filtered_storage.get('cookies', []):
				cookie = dict(raw_cookie)
				if cookie.get('expires') in {0, 0.0, -1, -1.0}:
					cookie.pop('expires', None)
				cookies.append(cookie)
			if cookies:
				await asyncio.wait_for(
					root_client.send.Storage.setCookies(params={'cookies': cookies, 'browserContextId': context_id}),
					timeout=8.0,
				)

			created_target = await asyncio.wait_for(
				root_client.send.Target.createTarget(params={'url': 'about:blank', 'browserContextId': context_id}),
				timeout=8.0,
			)
			target_id = created_target.get('targetId')
			if not target_id:
				raise QAIsolatedContextUnavailable('CDP did not create an isolated exploration target')
			isolated_target_id = target_id
			cdp_session = await self.browser_session.get_or_create_cdp_session(target_id, focus=True)
			local_storage, session_storage = self._qa_storage_maps(filtered_storage, root_origin)
			restore_script = f"""(() => {{
				if (window.location.origin !== {json.dumps(root_origin)}) return;
				const localItems = {json.dumps(local_storage)};
				const sessionItems = {json.dumps(session_storage)};
				for (const [key, value] of Object.entries(localItems)) localStorage.setItem(key, value);
				for (const [key, value] of Object.entries(sessionItems)) sessionStorage.setItem(key, value);
			}})();"""
			await asyncio.wait_for(
				cdp_session.cdp_client.send.Page.addScriptToEvaluateOnNewDocument(
					params={'source': restore_script}, session_id=cdp_session.session_id
				),
				timeout=8.0,
			)
			await asyncio.wait_for(
				cdp_session.cdp_client.send.Page.navigate(
					params={'url': self._qa_scope.root_url}, session_id=cdp_session.session_id
				),
				timeout=8.0,
			)
			context_ready = True
			return await self._compile_qa_case_from_exploration_page(compiler=compiler, draft=draft)
		except QAIsolatedContextUnavailable:
			raise
		except Exception as exc:
			if context_ready:
				raise
			raise QAIsolatedContextUnavailable(f'CDP isolated-context setup failed: {exc}') from exc
		finally:
			try:
				current_tabs = await self.browser_session.get_tabs()
				if original_target_id in {tab.target_id for tab in current_tabs}:
					switch_event = self.browser_session.event_bus.dispatch(SwitchTabEvent(target_id=original_target_id))
					await switch_event
					await switch_event.event_result(raise_if_any=False, raise_if_none=False)
			finally:
				if context_id:
					try:
						await asyncio.wait_for(
							root_client.send.Target.disposeBrowserContext(params={'browserContextId': context_id}), timeout=8.0
						)
					except Exception as exc:
						self._qa_warnings.append(f'Could not dispose isolated QA exploration context: {exc}')
				if isolated_target_id:
					for _ in range(20):
						if isolated_target_id not in {tab.target_id for tab in await self.browser_session.get_tabs()}:
							break
						await asyncio.sleep(0.05)

	async def _explore_and_complete_qa_case(self, *, compiler: QATaskCompiler, draft: WebUITestCaseDraft) -> WebUITestCase:
		"""Gather passive evidence in a disposable context, with snapshot fallback."""

		assert self._qa_scope is not None
		tabs_before = await self.browser_session.get_tabs()
		original_tab_ids = {tab.target_id for tab in tabs_before}
		original_target_id = self.browser_session.agent_focus_target_id
		if not original_target_id:
			raise QAExplorationStateError('No focused page is available for QA exploration')
		storage_state = await self.browser_session._cdp_get_storage_state()
		root_origin = self._qa_root_origin()

		try:
			compiled = await self._explore_in_isolated_browser_context(
				compiler=compiler,
				draft=draft,
				storage_state=storage_state,
				original_target_id=original_target_id,
				root_origin=root_origin,
			)
			# The disposable context owns its navigation. The formal browser must now
			# perform its single runner-owned root navigation before business steps.
			await self.browser_session.navigate_to(self._qa_scope.root_url, new_tab=False)
			await self._wait_for_qa_page_stability()
			self.logger.info('🧪 Completed read-only discovery in an isolated CDP BrowserContext.')
			return compiled
		except QAIsolatedContextUnavailable as exc:
			self._qa_warnings.append(f'Isolated exploration unavailable; used snapshot restoration fallback: {exc}')

		compiled_case: WebUITestCase | None = None
		primary_error: Exception | None = None
		try:
			await self.browser_session.navigate_to(self._qa_scope.root_url, new_tab=True)
			compiled_case = await self._compile_qa_case_from_exploration_page(compiler=compiler, draft=draft)
		except Exception as exc:
			primary_error = exc
		finally:
			try:
				await self._restore_qa_exploration_state(
					storage_state=storage_state,
					original_target_id=original_target_id,
					original_tab_ids=original_tab_ids,
					root_origin=root_origin,
				)
			except Exception:
				# Restoration uncertainty outranks compiler errors because continuing could
				# create false product failures from contaminated browser state.
				raise
		if primary_error is not None:
			raise primary_error
		assert compiled_case is not None
		return compiled_case

	async def _prepare_qa_case(self, draft: WebUITestCaseDraft) -> None:
		"""Compile missing expectations and establish a clean formal baseline."""

		assert self._qa_scope is not None
		compiler = QATaskCompiler(self.judge_llm)
		try:
			if draft.needs_exploration:
				self._qa_phase = 'exploration'
				self._qa_test_case = await self._explore_and_complete_qa_case(compiler=compiler, draft=draft)
			else:
				self._qa_test_case = await compiler.complete_with_discovery(
					draft=draft,
					scope=self._qa_scope,
					discovered_url=self._qa_scope.root_url,
					discovered_title='',
					discovered_dom='',
				)
				await self.browser_session.navigate_to(self._qa_scope.root_url, new_tab=False)
				await self._wait_for_qa_page_stability()
		finally:
			self._qa_llm_call_count += compiler.call_count

		self._qa_compiled_task = self._qa_original_task
		self._apply_qa_execution_limits()
		await self._notify_qa_plan_callback(self._qa_plan_snapshot_from_case(self._qa_test_case))
		self._log_qa_test_case_table('📋 Structured QA test case')
		await self._start_qa_execution()

	async def _prepare_reused_qa_case(self) -> None:
		"""Restore the root page and execute an already validated test case without recompilation."""

		assert self._qa_scope is not None
		assert self._qa_test_case is not None
		self._apply_qa_execution_limits()
		self._log_qa_test_case_table('♻️ Reused structured QA test case')
		await self.browser_session.navigate_to(self._qa_scope.root_url, new_tab=False)
		await self._wait_for_qa_page_stability()
		await self._start_qa_execution()

	def _log_qa_test_case_table(self, label: str) -> None:
		"""Log the compiled table without exposing values registered as sensitive data."""

		assert self._qa_test_case is not None
		table = self._qa_test_case.to_markdown_table()
		sensitive_values = collect_sensitive_data_values(self.sensitive_data)
		if sensitive_values:
			table = redact_sensitive_string(table, sensitive_values)
		self.logger.info(f'{label}:\n{table}')

	def _apply_qa_execution_limits(self) -> None:
		"""Apply runner-owned execution limits after LLM QA compilation."""

		if self._qa_test_case is None or self.settings.qa_step_limit is None:
			return
		limit = self.settings.qa_step_limit
		total_steps = len(self._qa_test_case.steps)
		if total_steps <= limit:
			return
		self._qa_test_case = self._qa_test_case.model_copy(
			update={'steps': self._qa_test_case.steps[:limit]},
			deep=True,
		)
		self._qa_warnings.append(f'QA execution limited to first {limit} of {total_steps} compiled business steps.')

	async def _start_qa_execution(self) -> None:
		"""Initialize per-run QA execution state after compilation or cache reuse."""

		assert self._qa_test_case is not None
		self._qa_phase = 'execution'
		self._qa_current_step_index = 0
		self._qa_step_retry_count = 0
		self._qa_step_results = []
		self._qa_current_attempt_receipts = []
		self._qa_step_history_start = len(self.history.history)
		self._qa_evidence_cursor = self._qa_evidence_monitor.cursor() if self._qa_evidence_monitor else None
		self._qa_before_snapshot = await self._capture_qa_snapshot('preconditions_or_step_1_before')
		if self._qa_test_case.preconditions:
			available_sensitive_refs = self._available_sensitive_reference_names()
			missing_refs = sorted(
				{
					ref
					for item in self._qa_test_case.preconditions
					if item.required
					for ref in item.sensitive_refs
					if ref not in available_sensitive_refs
				}
			)
			if missing_refs:
				self._qa_precondition_results = [
					QAPreconditionResult(
						precondition=item,
						status=(
							PreconditionStatus.BLOCKED
							if any(ref in missing_refs for ref in item.sensitive_refs)
							else PreconditionStatus.NOT_CHECKED
						),
						reason=f'Missing sensitive-data references: {missing_refs}',
					)
					for item in self._qa_test_case.preconditions
				]
				self._finalize_qa_result(
					status=QARunStatus.BLOCKED,
					failure_origin=FailureOrigin.ENVIRONMENT,
					failure_code=FailureCode.ENV_AUTH_REQUIRED,
					summary=f'BLOCKED: required sensitive data is missing: {missing_refs}',
				)
				return

			self._qa_running_preconditions = True
			self._qa_phase = 'preconditions'
			ensure_mode = any(item.mode == PreconditionMode.ENSURE for item in self._qa_test_case.preconditions)
			self._qa_precondition_step = WebUITestStep(
				step_id='__preconditions__',
				instruction='; '.join(item.description for item in self._qa_test_case.preconditions),
				expected_result='Every required precondition is satisfied before business-step execution.',
				expectation_source=ExpectationSource.EXPLICIT,
				source_evidence=[item.description for item in self._qa_test_case.preconditions],
				operation_kind=StepOperationKind.SUBMIT if ensure_mode else StepOperationKind.OBSERVE,
				side_effect_level=SideEffectLevel.REVERSIBLE if ensure_mode else SideEffectLevel.NONE,
			)
			self._message_manager.add_new_task(self._qa_preconditions_prompt(self._qa_precondition_step))
			return

		self._message_manager.add_new_task(self._qa_step_prompt(self._qa_test_case.steps[0]))

	@staticmethod
	def _qa_effective_operation(step: WebUITestStep, action_names: list[str]) -> StepOperationKind:
		if step.operation_kind != StepOperationKind.OTHER:
			return step.operation_kind
		if any(name in {'click', 'click_element', 'click_element_by_index'} for name in action_names):
			return StepOperationKind.CLICK
		if any(name in {'input', 'input_text', 'type'} for name in action_names):
			return StepOperationKind.INPUT
		if any(name in {'navigate', 'go_to_url', 'open_tab'} for name in action_names):
			return StepOperationKind.NAVIGATE
		if not action_names:
			return StepOperationKind.OBSERVE
		return StepOperationKind.OTHER

	@staticmethod
	def _qa_selected_element_matches(step: WebUITestStep, selected: dict[str, Any] | None) -> bool | None:
		"""Conservatively match an interacted element to the business instruction."""

		if selected is None:
			return None
		candidate_values = [
			str(selected.get('ax_name') or ''),
			str(selected.get('node_value') or ''),
		]
		attributes = selected.get('attributes') or {}
		for key in ('aria-label', 'title', 'name', 'placeholder', 'value'):
			candidate_values.append(str(attributes.get(key) or ''))
		instruction = re.sub(r'\s+', '', step.instruction).casefold()
		labels = [re.sub(r'\s+', '', value).casefold() for value in candidate_values if value.strip()]
		if not labels:
			return None
		return any(label in instruction or instruction in label for label in labels if len(label) >= 1)

	def _build_action_receipt(
		self,
		*,
		step: WebUITestStep,
		before: BrowserEvidenceSnapshot,
		after: BrowserEvidenceSnapshot,
		action_results: list[dict[str, Any]],
		action_names: list[str],
		selected_element: dict[str, Any] | None,
		input_values: list[str],
		side_effect_uncertain: bool,
	) -> tuple[ActionReceipt, EvidenceArtifact]:
		if selected_element is not None:
			selected_element = json.loads(self._redact_qa_text(json.dumps(selected_element, ensure_ascii=False)))
		operation = self._qa_effective_operation(step, action_names)
		errors = [str(result.get('error')) for result in action_results if result.get('error')]
		tool_succeeded = not errors and (bool(action_results) or operation == StepOperationKind.OBSERVE)
		tool_target_proofs: list[ActionTargetProof] = []
		tool_expectation_proofs: list[ActionExpectationProof] = []
		for result in action_results:
			if result.get('error'):
				continue
			metadata = result.get('metadata')
			if not isinstance(metadata, dict):
				continue
			if 'qa_target_proof' in metadata:
				try:
					tool_target_proofs.append(ActionTargetProof.model_validate(metadata['qa_target_proof']))
				except ValidationError as exc:
					logger.warning('Ignoring malformed qa_target_proof from a custom action: %s', exc)
			if 'qa_expectation_proof' in metadata:
				try:
					tool_expectation_proofs.append(ActionExpectationProof.model_validate(metadata['qa_expectation_proof']))
				except ValidationError as exc:
					logger.warning('Ignoring malformed qa_expectation_proof from a custom action: %s', exc)
		if operation in {StepOperationKind.OBSERVE, StepOperationKind.NAVIGATE}:
			target_matched = True
		elif tool_target_proofs:
			target_matched = all(proof.target_matched for proof in tool_target_proofs)
		else:
			target_matched = self._qa_selected_element_matches(step, selected_element)
		state_changed = before.url != after.url or before.dom_summary != after.dom_summary
		related_network = [
			artifact
			for artifact in after.artifacts
			if artifact.kind == EvidenceKind.NETWORK and artifact.metadata.get('request_id')
		]
		confirmed_request = any(
			str(artifact.metadata.get('method', 'GET')).upper() in {'POST', 'PUT', 'PATCH', 'DELETE'}
			and isinstance(artifact.metadata.get('status'), int)
			and 200 <= int(artifact.metadata['status']) < 400
			for artifact in related_network
		)
		if (
			step.side_effect_level == SideEffectLevel.IRREVERSIBLE
			and bool(action_names)
			and not (state_changed or confirmed_request)
		):
			side_effect_uncertain = True

		if side_effect_uncertain:
			status = ActionCompletionStatus.UNCERTAIN
			reasoning = 'A side-effecting action may have completed and cannot be safely replayed.'
		elif errors:
			status = ActionCompletionStatus.NOT_COMPLETED
			reasoning = f'Tool execution failed: {errors[0]}'
		elif operation == StepOperationKind.OBSERVE and after.dom_summary.strip():
			status = ActionCompletionStatus.COMPLETED
			reasoning = 'The observation step captured objective page evidence.'
		elif not tool_succeeded:
			status = ActionCompletionStatus.NOT_COMPLETED
			reasoning = 'No successful tool action was recorded for the business step.'
		elif target_matched is False:
			status = ActionCompletionStatus.COMPLETED
			reasoning = (
				'The tool action was dispatched successfully; element-level target metadata did not match the '
				'business step, so final judgement must use observable browser facts.'
			)
		elif target_matched is None and operation in {
			StepOperationKind.CLICK,
			StepOperationKind.INPUT,
			StepOperationKind.SUBMIT,
		}:
			status = ActionCompletionStatus.COMPLETED
			reasoning = (
				'The tool action was dispatched successfully; element-level target proof is unavailable, so final '
				'judgement must use observable browser facts.'
			)
		else:
			status = ActionCompletionStatus.COMPLETED
			reasoning = 'The intended action was dispatched successfully to the verified target.'

		action_summary = json.dumps(
			{
				'operation_kind': operation.value,
				'action_names': action_names,
				'input_values': input_values,
				'tool_succeeded': tool_succeeded,
				'target_matched': target_matched,
				'tool_target_proofs': [proof.model_dump(mode='json') for proof in tool_target_proofs],
				'tool_expectation_proofs': [proof.model_dump(mode='json') for proof in tool_expectation_proofs],
				'side_effect_uncertain': side_effect_uncertain,
				'errors': errors,
			},
			ensure_ascii=False,
		)
		artifact = self._create_qa_artifact(
			kind=EvidenceKind.ACTION,
			summary=action_summary,
			url=after.url,
			metadata={
				'selected_element': selected_element or {},
				'tool_target_proofs': [proof.model_dump(mode='json') for proof in tool_target_proofs],
				'tool_expectation_proofs': [proof.model_dump(mode='json') for proof in tool_expectation_proofs],
			},
		)
		related_requests = related_network
		receipt = ActionReceipt(
			status=status,
			operation_kind=operation,
			tool_succeeded=tool_succeeded,
			target_matched=target_matched,
			selected_element=selected_element,
			input_values=input_values,
			state_changed=state_changed,
			side_effect_uncertain=side_effect_uncertain,
			action_names=action_names,
			related_request_ids=[str(item.metadata['request_id']) for item in related_requests],
			evidence_ids=[artifact.evidence_id, *(item.evidence_id for item in related_requests)],
			reasoning=reasoning,
		)
		return receipt, artifact

	async def _build_qa_evidence(self, step: WebUITestStep) -> StepEvidence:
		assert self._qa_before_snapshot is not None
		after = await self._capture_qa_snapshot(
			f'step_{self._qa_current_step_index + 1}_attempt_{self._qa_step_retry_count + 1}_after'
		)
		action_results: list[dict[str, Any]] = []
		action_names: list[str] = []
		input_values: list[str] = []
		selected_element: dict[str, Any] | None = None
		for history_item in self.history.history[self._qa_step_history_start :]:
			if history_item.model_output is not None:
				for action in history_item.model_output.action:
					action_name = self._qa_replay_action_name(action)
					action_names.append(action_name)
					if action_name in {'input', 'input_text', 'type'}:
						action_payload = action.model_dump(exclude_none=True, mode='json').get(action_name, {})
						if isinstance(action_payload, dict):
							for key in ('text', 'value'):
								if key in action_payload:
									input_values.append(self._redact_qa_text(str(action_payload[key])))
			for interacted in history_item.state.interacted_element or []:
				if interacted is not None:
					selected_element = interacted.to_dict()
			for result in history_item.result:
				# ``finish_test_step`` is an executor claim, not objective evidence for the
				# independent judge. Keep only tool/browser action outcomes here.
				if result.metadata and 'qa_finish_test_step' in result.metadata:
					continue
				raw_result = result.model_dump(exclude_none=True, mode='json', exclude={'images'})
				action_results.append(json.loads(self._redact_qa_text(json.dumps(raw_result, ensure_ascii=False))))

		finish_metadata: dict[str, Any] = {}
		if self.history.history and self.history.history[-1].result:
			finish_metadata = self.history.history[-1].result[-1].metadata or {}
		finish_report = finish_metadata.get('qa_finish_test_step', {})
		side_effect_uncertain = bool(finish_report.get('side_effect_uncertain', False))
		receipt, action_artifact = self._build_action_receipt(
			step=step,
			before=self._qa_before_snapshot,
			after=after,
			action_results=action_results,
			action_names=[name for name in action_names if name != self._completion_action_name],
			selected_element=selected_element,
			input_values=input_values,
			side_effect_uncertain=side_effect_uncertain,
		)
		quality = EvidenceQuality.WEAK
		if receipt.status == ActionCompletionStatus.COMPLETED and after.dom_summary.strip():
			quality = EvidenceQuality.STRONG if receipt.target_matched is True else EvidenceQuality.MEDIUM
		return StepEvidence(
			before=self._qa_before_snapshot,
			after=after,
			action_results=action_results,
			action_receipt=receipt,
			artifacts=[action_artifact],
			executor_report=self._redact_qa_text(str(finish_report.get('actual_result') or '')) or None,
			side_effect_uncertain=side_effect_uncertain,
			evidence_quality=quality,
		)

	def _qa_mark_boundary_non_terminal(self, memory: str) -> None:
		"""Turn the executor's step-boundary action into ordinary history before continuing."""

		if self.history.history and self.history.history[-1].result:
			boundary = self.history.history[-1].result[-1]
			boundary.is_done = False
			boundary.success = None
		self.state.last_result = [ActionResult(long_term_memory=memory)]
		self.state.consecutive_failures = 0

	@staticmethod
	def _qa_run_status_for_step(status: QAStepStatus) -> tuple[QARunStatus, FailureOrigin]:
		mapping = {
			QAStepStatus.SUT_FAILED: (QARunStatus.SUT_FAILED, FailureOrigin.SUT),
			QAStepStatus.AGENT_FAILED: (QARunStatus.AGENT_FAILED, FailureOrigin.AGENT),
			QAStepStatus.BLOCKED: (QARunStatus.BLOCKED, FailureOrigin.ENVIRONMENT),
			QAStepStatus.INCONCLUSIVE: (QARunStatus.INCONCLUSIVE, FailureOrigin.UNKNOWN),
		}
		return mapping[status]

	@staticmethod
	def _qa_runtime_failure_status(exc: Exception, *, phase: str) -> tuple[QARunStatus, FailureOrigin]:
		"""Classify framework failures without ever turning them into SUT defects."""

		if isinstance(exc, ModelProviderError):
			return QARunStatus.AGENT_FAILED, FailureOrigin.AGENT
		if phase in {'browser_start', 'exploration'} or isinstance(exc, (BrowserError, ConnectionError, OSError)):
			return QARunStatus.BLOCKED, FailureOrigin.ENVIRONMENT
		message = f'{type(exc).__name__}: {exc}'.lower()
		browser_failure_markers = ('browser', 'chromium', 'cdp', 'websocket', 'target closed', 'network connection')
		if any(marker in message for marker in browser_failure_markers):
			return QARunStatus.BLOCKED, FailureOrigin.ENVIRONMENT
		return QARunStatus.AGENT_FAILED, FailureOrigin.AGENT

	@staticmethod
	def _qa_runtime_failure_code(exc: Exception, *, phase: str) -> FailureCode:
		"""Attach an auditable subtype without ever classifying framework errors as SUT defects."""

		message = f'{type(exc).__name__}: {exc}'.casefold()
		if 'captcha' in message:
			return FailureCode.ENV_CAPTCHA
		if any(marker in message for marker in ('out of scope', 'navigation policy', 'allowed_domains', 'prohibited_domains')):
			return FailureCode.ENV_NAVIGATION_POLICY
		if any(marker in message for marker in ('credential', 'authentication', 'login required', 'sign in')):
			return FailureCode.ENV_AUTH_REQUIRED
		if isinstance(exc, ModelProviderError):
			return FailureCode.AGENT_MODEL_FAILURE
		if phase == 'specification':
			return FailureCode.AGENT_COMPILER_ERROR
		if any(marker in message for marker in ('network', 'dns', 'connection refused', 'timed out')):
			return FailureCode.ENV_NETWORK
		if phase in {'browser_start', 'exploration'} or any(
			marker in message for marker in ('browser', 'chromium', 'cdp', 'websocket', 'target closed')
		):
			return FailureCode.ENV_BROWSER
		return FailureCode.AGENT_ACTION_ERROR

	def _finalize_qa_result(
		self,
		*,
		status: QARunStatus,
		failure_origin: FailureOrigin | None,
		summary: str,
		stopped_at_step: str | None = None,
		validation_errors: list[str] | None = None,
		failure_code: FailureCode | None = None,
	) -> None:
		if self._qa_test_case is not None:
			recorded_ids = {result.step.step_id for result in self._qa_step_results}
			for step in self._qa_test_case.steps:
				if step.step_id not in recorded_ids:
					self._qa_step_results.append(QAStepResult(step=step, status=QAStepStatus.NOT_RUN))

		if failure_code is None:
			for recorded in reversed(self._qa_step_results):
				if recorded.judgement is not None:
					failure_code = recorded.judgement.failure_code
					break
		failure_code = failure_code or FailureCode.NONE
		artifacts: list[EvidenceArtifact] = []
		seen_artifacts: set[str] = set()
		for recorded in self._qa_step_results:
			if recorded.evidence is None:
				continue
			for artifact in recorded.evidence.artifacts:
				if artifact.evidence_id not in seen_artifacts:
					artifacts.append(artifact)
					seen_artifacts.add(artifact.evidence_id)

		viewport = self.browser_profile.viewport
		self.history.qa_result = QARunResult(
			run_id=self._qa_run_id,
			status=status,
			failure_origin=failure_origin,
			failure_code=failure_code,
			test_case=self._qa_test_case,
			precondition_results=self._qa_precondition_results,
			step_results=self._qa_step_results,
			cleanup_results=self._qa_cleanup_results,
			stopped_at_step=stopped_at_step,
			summary=summary,
			validation_errors=validation_errors or [],
			warnings=self._qa_warnings,
			requested_mode=self._qa_requested_mode,
			effective_mode=self._qa_effective_mode,
			llm_call_count=self._qa_llm_call_count,
			environment={
				'root_url': self._qa_scope.root_url if self._qa_scope else None,
				'headless': self.browser_profile.headless,
				'viewport': viewport.model_dump() if viewport is not None and hasattr(viewport, 'model_dump') else viewport,
			},
			artifacts=artifacts,
			phase_timings=[
				QAPhaseTiming(phase=phase, elapsed_seconds=elapsed) for phase, elapsed in self._qa_phase_timings.items()
			],
		)
		if not self.history.history:
			self.history.add_item(
				AgentHistory(
					model_output=None,
					result=[ActionResult(is_done=True, success=None, extracted_content=summary, long_term_memory=summary)],
					state=BrowserStateHistory(
						url=self._qa_scope.root_url if self._qa_scope else '',
						title='',
						tabs=[],
						interacted_element=[],
						screenshot_path=None,
					),
					metadata=None,
				)
			)
		terminal_success: bool | None = None
		if status == QARunStatus.PASSED:
			terminal_success = True
		elif status == QARunStatus.SUT_FAILED:
			terminal_success = False
		if self.history.history and self.history.history[-1].result:
			terminal = self.history.history[-1].result[-1]
			terminal.is_done = True
			terminal.success = terminal_success
			terminal.extracted_content = summary

	async def _finalize_or_start_qa_cleanup(
		self,
		*,
		status: QARunStatus,
		failure_origin: FailureOrigin | None,
		summary: str,
		stopped_at_step: str | None = None,
		failure_code: FailureCode | None = None,
		after: BrowserEvidenceSnapshot,
	) -> bool:
		"""Run optional cleanup without allowing it to replace the business verdict."""

		assert self._qa_test_case is not None
		if not self._qa_test_case.cleanup_steps:
			self._finalize_qa_result(
				status=status,
				failure_origin=failure_origin,
				failure_code=failure_code,
				stopped_at_step=stopped_at_step,
				summary=summary,
			)
			return True

		self._qa_pending_business_outcome = {
			'status': status,
			'failure_origin': failure_origin,
			'failure_code': failure_code,
			'stopped_at_step': stopped_at_step,
			'summary': summary,
		}
		self._qa_running_cleanup = True
		self._qa_cleanup_index = 0
		self._qa_step_retry_count = 0
		self._qa_mark_boundary_non_terminal('Business verdict recorded; starting isolated cleanup.')
		self._qa_before_snapshot = after
		self._qa_evidence_cursor = self._qa_evidence_monitor.cursor() if self._qa_evidence_monitor else None
		self._qa_step_history_start = len(self.history.history)
		self._message_manager.add_new_task(self._qa_step_prompt(self._qa_test_case.cleanup_steps[0]))
		return False

	def _finalize_pending_business_outcome(self) -> None:
		"""Publish the stored business verdict after cleanup completes or fails."""

		if self._qa_pending_business_outcome is None:
			raise RuntimeError('No pending QA business outcome exists')
		outcome = self._qa_pending_business_outcome
		self._qa_pending_business_outcome = None
		self._qa_running_cleanup = False
		self._finalize_qa_result(**outcome)

	def _abort_qa_cleanup(self, reason: str) -> None:
		"""Record cleanup infrastructure failure while preserving the business outcome."""

		if not self._qa_running_cleanup or self._qa_pending_business_outcome is None:
			return
		assert self._qa_test_case is not None
		if self._qa_cleanup_index < len(self._qa_test_case.cleanup_steps):
			step = self._qa_test_case.cleanup_steps[self._qa_cleanup_index]
			self._qa_cleanup_results.append(QACleanupResult(step=step, status=QAStepStatus.AGENT_FAILED, reason=reason))
		self._qa_warnings.append(f'Cleanup aborted: {reason}')
		self._finalize_pending_business_outcome()

	async def _handle_qa_step_boundary(self) -> bool:
		"""Judge a submitted step. Return True only when the complete QA run is terminal."""

		assert self._qa_test_case is not None
		step = (
			self._qa_precondition_step
			if self._qa_running_preconditions and self._qa_precondition_step is not None
			else (
				self._qa_test_case.cleanup_steps[self._qa_cleanup_index]
				if self._qa_running_cleanup
				else self._qa_test_case.steps[self._qa_current_step_index]
			)
		)
		evidence = await self._build_qa_evidence(step)
		if evidence.action_receipt is not None:
			self._qa_current_attempt_receipts.append(evidence.action_receipt)
		review: ReviewRecord | None = None
		try:
			judgement = await judge_test_step(
				llm=self.judge_llm,
				step=step,
				evidence=evidence,
				action_receipt=evidence.action_receipt,
				on_llm_call=self._record_qa_llm_call,
				call_label='judge',
			)
		except Exception as exc:
			self.logger.error(f'Independent QA step judge failed: {exc}', exc_info=True)
			judgement = StepJudgement(
				action_status=ActionCompletionStatus.UNCERTAIN,
				expectation_status=ExpectationStatus.NOT_OBSERVABLE,
				status=QAStepStatus.AGENT_FAILED,
				failure_origin=FailureOrigin.AGENT,
				failure_code=FailureCode.AGENT_JUDGE_ERROR,
				reasoning=f'Independent judge failed: {type(exc).__name__}: {exc}',
				actual_result='No reliable judgement was produced.',
				evidence=[],
				retry_safe=False,
			)

		review_required = (
			not self._qa_running_preconditions
			and not self._qa_running_cleanup
			and (
				judgement.status == QAStepStatus.SUT_FAILED
				or (judgement.status == QAStepStatus.PASSED and judgement.confidence < 0.8)
			)
		)
		if review_required:
			review = ReviewRecord(
				primary_status=judgement.status,
				reviewer_model=getattr(self.review_llm, 'model', None),
				evidence_ids=judgement.evidence_ids,
			)
			try:
				secondary = await judge_test_step(
					llm=self.review_llm,
					step=step,
					evidence=evidence,
					action_receipt=evidence.action_receipt,
					on_llm_call=self._record_qa_llm_call,
					call_label='review',
				)
				review.secondary_status = secondary.status
				review.agreed = secondary.status == judgement.status
				review.reason = secondary.reasoning
				if not review.agreed:
					judgement = StepJudgement(
						action_status=evidence.action_receipt.status
						if evidence.action_receipt
						else ActionCompletionStatus.UNCERTAIN,
						expectation_status=ExpectationStatus.NOT_OBSERVABLE,
						status=QAStepStatus.INCONCLUSIVE,
						failure_origin=FailureOrigin.UNKNOWN,
						failure_code=FailureCode.UNKNOWN_CONFLICT,
						reasoning=(
							f'Independent risk review disagreed with the primary verdict: '
							f'{review.primary_status.value} vs {secondary.status.value}.'
						),
						actual_result='Independent Judges did not agree on a reliable product verdict.',
						evidence_ids=sorted(set(judgement.evidence_ids + secondary.evidence_ids)),
						confidence=min(judgement.confidence, secondary.confidence, 0.49),
					)
			except Exception as exc:
				review.reason = f'Review Judge failed: {type(exc).__name__}: {exc}'
				judgement = StepJudgement(
					action_status=evidence.action_receipt.status if evidence.action_receipt else ActionCompletionStatus.UNCERTAIN,
					expectation_status=ExpectationStatus.NOT_OBSERVABLE,
					status=QAStepStatus.AGENT_FAILED,
					failure_origin=FailureOrigin.AGENT,
					failure_code=FailureCode.AGENT_JUDGE_ERROR,
					reasoning=review.reason,
					actual_result='Required independent SUT review was not produced.',
					evidence_ids=judgement.evidence_ids,
				)

		if (
			judgement.status == QAStepStatus.AGENT_FAILED
			and judgement.retry_safe
			and not evidence.side_effect_uncertain
			and self._qa_step_retry_count < self.settings.max_agent_retries_per_step
		):
			self._qa_step_retry_count += 1
			self._qa_mark_boundary_non_terminal(
				f'Independent judge requested safe agent recovery attempt {self._qa_step_retry_count}: {judgement.reasoning}'
			)
			self._qa_before_snapshot = evidence.after
			self._qa_evidence_cursor = self._qa_evidence_monitor.cursor() if self._qa_evidence_monitor else None
			self._qa_step_history_start = len(self.history.history)
			self._message_manager.add_new_task(self._qa_step_prompt(step, retry_reason=judgement.reasoning))
			return False

		if self._qa_running_cleanup:
			self._qa_cleanup_results.append(
				QACleanupResult(
					step=step,
					status=judgement.status,
					reason=judgement.actual_result,
					evidence_ids=judgement.evidence_ids,
				)
			)
			if judgement.status != QAStepStatus.PASSED:
				self._qa_warnings.append(f'Cleanup {step.step_id} ended as {judgement.status.value}: {judgement.actual_result}')
			self._qa_cleanup_index += 1
			if self._qa_cleanup_index < len(self._qa_test_case.cleanup_steps):
				self._qa_mark_boundary_non_terminal(f'Cleanup {step.step_id} recorded; continuing cleanup.')
				self._qa_step_retry_count = 0
				self._qa_before_snapshot = evidence.after
				self._qa_evidence_cursor = self._qa_evidence_monitor.cursor() if self._qa_evidence_monitor else None
				self._qa_step_history_start = len(self.history.history)
				self._message_manager.add_new_task(self._qa_step_prompt(self._qa_test_case.cleanup_steps[self._qa_cleanup_index]))
				return False
			self._finalize_pending_business_outcome()
			return True

		if self._qa_running_preconditions:
			if judgement.status == QAStepStatus.PASSED:
				self._qa_precondition_results = [
					QAPreconditionResult(
						precondition=item,
						status=(
							PreconditionStatus.ENSURED if item.mode == PreconditionMode.ENSURE else PreconditionStatus.SATISFIED
						),
						reason=judgement.actual_result,
						evidence_ids=judgement.evidence_ids,
					)
					for item in self._qa_test_case.preconditions
				]
				self._qa_mark_boundary_non_terminal('All required QA preconditions were satisfied.')
				self._qa_running_preconditions = False
				self._qa_precondition_step = None
				self._qa_phase = 'execution'
				self._qa_step_retry_count = 0
				self._qa_current_attempt_receipts = []
				self._qa_before_snapshot = evidence.after
				self._qa_evidence_cursor = self._qa_evidence_monitor.cursor() if self._qa_evidence_monitor else None
				self._qa_step_history_start = len(self.history.history)
				self._message_manager.add_new_task(self._qa_step_prompt(self._qa_test_case.steps[0]))
				return False

			self._qa_precondition_results = [
				QAPreconditionResult(
					precondition=item,
					status=PreconditionStatus.BLOCKED,
					reason=judgement.actual_result,
					evidence_ids=judgement.evidence_ids,
				)
				for item in self._qa_test_case.preconditions
			]
			if judgement.status == QAStepStatus.AGENT_FAILED:
				status = QARunStatus.AGENT_FAILED
				origin = FailureOrigin.AGENT
				code = judgement.failure_code
			elif judgement.status == QAStepStatus.INCONCLUSIVE:
				status = QARunStatus.INCONCLUSIVE
				origin = FailureOrigin.UNKNOWN
				code = judgement.failure_code
			else:
				status = QARunStatus.BLOCKED
				origin = FailureOrigin.ENVIRONMENT
				code = FailureCode.ENV_TEST_DATA
			self._finalize_qa_result(
				status=status,
				failure_origin=origin,
				failure_code=code,
				summary=f'{status.value} during QA preconditions: {judgement.actual_result}',
			)
			return True

		step_result = QAStepResult(
			step=step,
			status=judgement.status,
			retry_count=self._qa_step_retry_count,
			judgement=judgement,
			review=review,
			evidence=evidence,
			attempt_receipts=[receipt.model_copy(deep=True) for receipt in self._qa_current_attempt_receipts],
		)
		self._qa_step_results.append(step_result)
		if judgement.status == QAStepStatus.PASSED and judgement.replay_assertions:
			self._qa_replay_history[step.step_id] = [
				item.model_copy(deep=True) for item in self.history.history[self._qa_step_history_start :]
			]
		if judgement.status == QAStepStatus.PASSED and self._qa_replay_fallback_active:
			self._qa_mark_boundary_non_terminal(f'QA replay fallback repaired {step.step_id}.')
			self._qa_replay_fallback_active = False
			self._qa_replay_fallback_completed = True
			return True

		if judgement.status != QAStepStatus.PASSED:
			run_status, failure_origin = self._qa_run_status_for_step(judgement.status)
			return await self._finalize_or_start_qa_cleanup(
				status=run_status,
				failure_origin=failure_origin,
				stopped_at_step=step.step_id,
				summary=f'{run_status.value} at {step.step_id}: {judgement.actual_result}',
				after=evidence.after,
			)

		self._qa_current_step_index += 1
		if self._qa_current_step_index >= len(self._qa_test_case.steps):
			return await self._finalize_or_start_qa_cleanup(
				status=QARunStatus.PASSED,
				failure_origin=FailureOrigin.NONE,
				summary=f'PASSED: all {len(self._qa_test_case.steps)} business-step expectations were met.',
				after=evidence.after,
			)

		self._qa_mark_boundary_non_terminal(f'QA step {step.step_id} passed independent judgement.')
		self._qa_step_retry_count = 0
		self._qa_current_attempt_receipts = []
		self._qa_before_snapshot = evidence.after
		self._qa_evidence_cursor = self._qa_evidence_monitor.cursor() if self._qa_evidence_monitor else None
		self._qa_step_history_start = len(self.history.history)
		next_step = self._qa_test_case.steps[self._qa_current_step_index]
		self._message_manager.add_new_task(self._qa_step_prompt(next_step))
		return False

	async def _notify_qa_done_callback(self) -> None:
		"""Invoke the public done callback exactly once for every terminal QA run."""

		if self._qa_done_callback_called or self.register_done_callback is None:
			return
		self._qa_done_callback_called = True
		if inspect.iscoroutinefunction(self.register_done_callback):
			await self.register_done_callback(self.history)
		else:
			self.register_done_callback(self.history)

	@staticmethod
	def _qa_plan_step_preconditions(step: Any) -> list[str]:
		return [item.description if hasattr(item, 'description') else str(item) for item in getattr(step, 'preconditions', [])]

	@classmethod
	def _qa_plan_snapshot_from_draft(cls, draft: WebUITestCaseDraft) -> QAPlanSnapshot:
		return QAPlanSnapshot(
			status='ready',
			preconditions=list(draft.preconditions),
			steps=[
				QAPlanStep(
					step_num=index,
					step_id=step.step_id,
					instruction=step.instruction,
					expected_result=step.expected_result,
					operation_kind=step.operation_kind,
					side_effect_level=step.side_effect_level,
					preconditions=cls._qa_plan_step_preconditions(step),
					source_evidence=list(step.source_evidence),
				)
				for index, step in enumerate(draft.steps, start=1)
			],
			needs_exploration=draft.needs_exploration,
		)

	@classmethod
	def _qa_plan_snapshot_from_case(cls, test_case: WebUITestCase) -> QAPlanSnapshot:
		return QAPlanSnapshot(
			status='final',
			preconditions=[precondition.description for precondition in test_case.preconditions],
			steps=[
				QAPlanStep(
					step_num=index,
					step_id=step.step_id,
					instruction=step.instruction,
					expected_result=step.expected_result,
					operation_kind=step.operation_kind,
					side_effect_level=step.side_effect_level,
					preconditions=cls._qa_plan_step_preconditions(step),
					source_evidence=list(step.source_evidence),
				)
				for index, step in enumerate(test_case.steps, start=1)
			],
			needs_exploration=False,
		)

	@staticmethod
	def _qa_plan_failed_snapshot(error_message: str) -> QAPlanSnapshot:
		return QAPlanSnapshot(status='failed', error_message=error_message)

	async def _notify_qa_plan_callback(self, snapshot: QAPlanSnapshot) -> None:
		"""Invoke the public QA plan callback without letting callback failures fail the QA run."""

		callback = self.register_qa_plan_callback
		if callback is None:
			return
		try:
			result = callback(snapshot)
			if inspect.isawaitable(result):
				await result
		except Exception as exc:
			self.logger.warning(
				'QA plan callback failed for status %s: %s: %s',
				snapshot.status,
				type(exc).__name__,
				exc,
			)

	@observe(name='agent.run', ignore_input=True, ignore_output=True)
	@time_execution_async('--run')
	async def run(
		self,
		max_steps: int = 500,
		on_step_start: AgentHookFunc | None = None,
		on_step_end: AgentHookFunc | None = None,
	) -> AgentHistoryList[AgentStructuredOutput]:
		"""Compile and execute a Web UI QA case with independent per-step judgement."""

		loop = asyncio.get_event_loop()
		agent_run_error: str | None = None  # Initialize error tracking variable
		self._force_exit_telemetry_logged = False  # ADDED: Flag for custom telemetry on force exit
		should_delay_close = False

		# Set up the  signal handler with callbacks specific to this agent
		from browser_use.utils import SignalHandler

		# Define the custom exit callback function for second CTRL+C
		def on_force_exit_log_telemetry():
			self._log_agent_event(max_steps=max_steps, agent_run_error='SIGINT: Cancelled by user')
			# NEW: Call the flush method on the telemetry instance
			if hasattr(self, 'telemetry') and self.telemetry:
				self.telemetry.flush()
			self._force_exit_telemetry_logged = True  # Set the flag

		signal_handler = SignalHandler(
			loop=loop,
			pause_callback=self.pause,
			resume_callback=self.resume,
			custom_exit_callback=on_force_exit_log_telemetry,  # Pass the new telemetrycallback
			exit_on_second_int=True,
			disabled=not self.enable_signal_handler,
		)
		signal_handler.register()

		try:
			await self._log_agent_run()

			# Initialize timing before specification validation because final task
			# events are emitted for INVALID_SPEC runs as well.
			self._session_start_time = time.time()
			self._task_start_time = self._session_start_time

			if self._qa_scope_error:
				await self._notify_qa_plan_callback(self._qa_plan_failed_snapshot(self._qa_scope_error))
				self._finalize_qa_result(
					status=QARunStatus.INVALID_SPEC,
					failure_origin=None,
					summary=f'INVALID_SPEC: {self._qa_scope_error}',
					validation_errors=[self._qa_scope_error],
				)
				self.history.usage = await self.token_cost_service.get_usage_summary()
				await self._notify_qa_done_callback()
				return self.history

			reuse_cached_case = bool(
				self.settings.reuse_compiled_test_case
				and self._qa_test_case is not None
				and self._qa_compiled_task is not None
				and self._qa_original_task.strip() == self._qa_compiled_task.strip()
			)
			if reuse_cached_case:
				self._qa_test_case_draft = None
				self.logger.info('♻️ Skipping Task compilation and discovery LLM calls for this repeated QA run.')
				assert self._qa_test_case is not None
				await self._notify_qa_plan_callback(self._qa_plan_snapshot_from_case(self._qa_test_case))
			else:
				# Stage 1 is browser-independent. A malformed/uncompilable specification
				# must not launch Chromium or mutate any page state.
				try:
					compiler = QATaskCompiler(self.judge_llm)
					try:
						await self._notify_qa_plan_callback(QAPlanSnapshot(status='generating'))
						self._qa_test_case_draft = await compiler.extract_requirements(
							task=self._qa_original_task,
							ground_truth=self.settings.ground_truth,
						)
						await self._notify_qa_plan_callback(self._qa_plan_snapshot_from_draft(self._qa_test_case_draft))
					finally:
						self._qa_llm_call_count += compiler.call_count
				except (ValidationError, ValueError) as exc:
					error_message = f'test structure could not be generated: {type(exc).__name__}: {exc}'
					await self._notify_qa_plan_callback(self._qa_plan_failed_snapshot(error_message))
					self._finalize_qa_result(
						status=QARunStatus.INVALID_SPEC,
						failure_origin=None,
						summary=f'INVALID_SPEC: {error_message}',
						validation_errors=[str(exc)],
					)
					self.history.usage = await self.token_cost_service.get_usage_summary()
					await self._notify_qa_done_callback()
					return self.history
				except Exception as exc:
					status, origin = self._qa_runtime_failure_status(exc, phase='specification')
					error_message = f'QA specification compilation failed: {type(exc).__name__}: {exc}'
					await self._notify_qa_plan_callback(self._qa_plan_failed_snapshot(error_message))
					self._finalize_qa_result(
						status=status,
						failure_origin=origin,
						failure_code=self._qa_runtime_failure_code(exc, phase='specification'),
						summary=f'{status.value} during {error_message}',
					)
					self.history.usage = await self.token_cost_service.get_usage_summary()
					await self._notify_qa_done_callback()
					return self.history

			self.logger.debug(
				f'🔧 Agent setup: Agent Session ID {self.session_id[-4:]}, Task ID {self.task_id[-4:]}, Browser Session ID {self.browser_session.id[-4:] if self.browser_session else "None"} {"(connecting via CDP)" if (self.browser_session and self.browser_session.cdp_url) else "(launching local browser)"}'
			)

			# Only dispatch session events if this is the first run
			if not self.state.session_initialized:
				self.logger.debug('📡 Dispatching CreateAgentSessionEvent...')
				# Emit CreateAgentSessionEvent at the START of run()
				self.eventbus.dispatch(CreateAgentSessionEvent.from_agent(self))

				self.state.session_initialized = True

			self.logger.debug('📡 Dispatching CreateAgentTaskEvent...')
			# Emit CreateAgentTaskEvent at the START of run()
			self.eventbus.dispatch(CreateAgentTaskEvent.from_agent(self))

			# Log startup message on first step (only if we haven't already done steps)
			self._log_first_step_startup()
			# Start browser session and attach watchdogs
			self._qa_phase = 'browser_start'
			self._ensure_browser_session_can_restart()
			await self.browser_session.start()
			await self._restore_cached_qa_login_state()
			self._qa_evidence_monitor = QAEvidenceMonitor(self.browser_session, self._qa_scope)
			await self._qa_evidence_monitor.start()
			if self._demo_mode_enabled:
				await self._demo_mode_log(f'Started task: {self.task}', 'info', {'tag': 'task'})
				await self._demo_mode_log(
					'Demo mode active - follow the side panel for live thoughts and actions.',
					'info',
					{'tag': 'status'},
				)

			# Register skills as actions if SkillService is configured
			await self._register_skills_as_actions()

			# Normally there was no try catch here but the callback can raise an InterruptedError.
			# Wrap with step_timeout so initial actions (usually a single URL navigate) can't
			# hang indefinitely on a silent CDP WebSocket — without this the agent would take
			# zero steps and return with an empty history while any outer watchdog waits.
			try:
				await asyncio.wait_for(
					self._execute_initial_actions(),
					timeout=self.settings.step_timeout,
				)
			except InterruptedError:
				pass
			except TimeoutError:
				initial_timeout_msg = (
					f'Initial actions timed out after {self.settings.step_timeout}s '
					f'(browser may be unresponsive). Proceeding to main execution loop.'
				)
				self.logger.error(f'⏰ {initial_timeout_msg}')
				self.state.last_result = [ActionResult(error=initial_timeout_msg)]
				self.state.consecutive_failures += 1
			except Exception as e:
				raise e

			try:
				if reuse_cached_case:
					await self._prepare_reused_qa_case()
				else:
					assert self._qa_test_case_draft is not None
					await self._prepare_qa_case(self._qa_test_case_draft)
				if self.history.qa_result is not None:
					self.history.usage = await self.token_cost_service.get_usage_summary()
					await self._notify_qa_done_callback()
					return self.history
			except QAExplorationStateError as exc:
				status = QARunStatus.INCONCLUSIVE if exc.inconclusive else QARunStatus.BLOCKED
				origin = FailureOrigin.UNKNOWN if exc.inconclusive else FailureOrigin.ENVIRONMENT
				self._finalize_qa_result(
					status=status,
					failure_origin=origin,
					summary=f'{status.value} during QA exploration restoration: {exc}',
				)
				self.history.usage = await self.token_cost_service.get_usage_summary()
				await self._notify_qa_done_callback()
				return self.history
			except (ValidationError, ValueError) as exc:
				await self._notify_qa_plan_callback(self._qa_plan_failed_snapshot(f'{type(exc).__name__}: {exc}'))
				self._finalize_qa_result(
					status=QARunStatus.INVALID_SPEC,
					failure_origin=None,
					summary=f'INVALID_SPEC: {exc}',
					validation_errors=[str(exc)],
				)
				self.history.usage = await self.token_cost_service.get_usage_summary()
				await self._notify_qa_done_callback()
				return self.history
			except Exception as exc:
				status, origin = self._qa_runtime_failure_status(exc, phase=self._qa_phase)
				await self._notify_qa_plan_callback(self._qa_plan_failed_snapshot(f'{type(exc).__name__}: {exc}'))
				self._finalize_qa_result(
					status=status,
					failure_origin=origin,
					failure_code=self._qa_runtime_failure_code(exc, phase=self._qa_phase),
					summary=f'{status.value} during QA discovery/compilation: {type(exc).__name__}: {exc}',
				)
				self.history.usage = await self.token_cost_service.get_usage_summary()
				await self._notify_qa_done_callback()
				return self.history

			self.logger.debug(
				f'🔄 Starting main execution loop with max {max_steps} steps (currently at step {self.state.n_steps})...'
			)
			while self.state.n_steps <= max_steps:
				current_step = self.state.n_steps - 1  # Convert to 0-indexed for step_info

				# Use the consolidated pause state management
				if self.state.paused:
					self.logger.debug(f'⏸️ Step {self.state.n_steps}: Agent paused, waiting to resume...')
					await self._external_pause_event.wait()
					signal_handler.reset()

				# Check if we should stop due to too many failures, if final_response_after_failure is True, we try one last time
				if (self.state.consecutive_failures) >= self.settings.max_failures + int(
					self.settings.final_response_after_failure
				):
					self.logger.error(f'❌ Stopping due to {self.settings.max_failures} consecutive failures')
					agent_run_error = f'Stopped due to {self.settings.max_failures} consecutive failures'
					break

				# Check control flags before each step
				if self.state.stopped:
					self.logger.info('🛑 Agent stopped')
					agent_run_error = 'Agent stopped programmatically'
					break

				step_info = AgentStepInfo(step_number=current_step, max_steps=max_steps)
				is_done = await self._execute_step(current_step, max_steps, step_info, on_step_start, on_step_end)

				if is_done:
					qa_terminal = await self._handle_qa_step_boundary()
					if not qa_terminal:
						continue

					# The independent judge has produced a terminal case result.
					if self._demo_mode_enabled and self.history.history:
						final_result_text = self.history.final_result() or 'QA run completed'
						await self._demo_mode_log(f'Final Result: {final_result_text}', 'success', {'tag': 'task'})
					await self._notify_qa_done_callback()

					should_delay_close = True
					break
			else:
				agent_run_error = 'Failed to complete task in maximum steps'

				self.history.add_item(
					AgentHistory(
						model_output=None,
						result=[ActionResult(error=agent_run_error, include_in_memory=True)],
						state=BrowserStateHistory(
							url='',
							title='',
							tabs=[],
							interacted_element=[],
							screenshot_path=None,
						),
						metadata=None,
					)
				)

				self.logger.info(f'❌ {agent_run_error}')

			if self.history.qa_result is None:
				if self._qa_pending_business_outcome is not None:
					self._abort_qa_cleanup(agent_run_error or 'Cleanup execution ended without a result')
				else:
					status = QARunStatus.BLOCKED if self.state.stopped else QARunStatus.AGENT_FAILED
					origin = FailureOrigin.ENVIRONMENT if status == QARunStatus.BLOCKED else FailureOrigin.AGENT
					self._finalize_qa_result(
						status=status,
						failure_origin=origin,
						summary=f'{status.value}: {agent_run_error or "execution ended without a QA verdict"}',
						stopped_at_step=(
							self._qa_test_case.steps[self._qa_current_step_index].step_id
							if self._qa_test_case and self._qa_current_step_index < len(self._qa_test_case.steps)
							else None
						),
					)

			self.history.usage = await self.token_cost_service.get_usage_summary()

			# set the model output schema and call it on the fly
			if self.history._output_model_schema is None and self.output_model_schema is not None:
				self.history._output_model_schema = self.output_model_schema

			await self._notify_qa_done_callback()
			return self.history

		except KeyboardInterrupt:
			# Already handled by our signal handler, but catch any direct KeyboardInterrupt as well
			self.logger.debug('Got KeyboardInterrupt during execution, returning current history')
			agent_run_error = 'KeyboardInterrupt'

			if self.history.qa_result is None:
				if self._qa_pending_business_outcome is not None:
					self._abort_qa_cleanup('Cleanup interrupted by the operator')
				else:
					self._finalize_qa_result(
						status=QARunStatus.BLOCKED,
						failure_origin=FailureOrigin.ENVIRONMENT,
						summary='BLOCKED: execution interrupted by the operator',
						stopped_at_step=(
							self._qa_test_case.steps[self._qa_current_step_index].step_id
							if self._qa_test_case and self._qa_current_step_index < len(self._qa_test_case.steps)
							else None
						),
					)

			self.history.usage = await self.token_cost_service.get_usage_summary()

			await self._notify_qa_done_callback()
			return self.history

		except Exception as e:
			self.logger.error(f'Agent run failed with exception: {e}', exc_info=True)
			agent_run_error = str(e)
			if self.history.qa_result is None:
				if self._qa_pending_business_outcome is not None:
					self._abort_qa_cleanup(f'{type(e).__name__}: {e}')
				else:
					status, origin = self._qa_runtime_failure_status(e, phase=self._qa_phase)
					self._finalize_qa_result(
						status=status,
						failure_origin=origin,
						failure_code=self._qa_runtime_failure_code(e, phase=self._qa_phase),
						summary=f'{status.value}: {type(e).__name__}: {e}',
						stopped_at_step=(
							self._qa_test_case.steps[self._qa_current_step_index].step_id
							if self._qa_test_case and self._qa_current_step_index < len(self._qa_test_case.steps)
							else None
						),
					)
			self.history.usage = await self.token_cost_service.get_usage_summary()
			await self._notify_qa_done_callback()
			return self.history

		finally:
			if should_delay_close and self._demo_mode_enabled and agent_run_error is None:
				await asyncio.sleep(30)
			if agent_run_error:
				await self._demo_mode_log(f'Agent stopped: {agent_run_error}', 'error', {'tag': 'run'})
			# Log token usage summary
			await self.token_cost_service.log_usage_summary()

			# Unregister signal handlers before cleanup
			signal_handler.unregister()

			if not self._force_exit_telemetry_logged:  # MODIFIED: Check the flag
				try:
					self._log_agent_event(max_steps=max_steps, agent_run_error=agent_run_error)
				except Exception as log_e:  # Catch potential errors during logging itself
					self.logger.error(f'Failed to log telemetry event: {log_e}', exc_info=True)
			else:
				# ADDED: Info message when custom telemetry for SIGINT was already logged
				self.logger.debug('Telemetry for force exit (SIGINT) was logged by custom exit callback.')

			# NOTE: CreateAgentSessionEvent and CreateAgentTaskEvent are now emitted at the START of run()
			# to match backend requirements for CREATE events to be fired when entities are created,
			# not when they are completed

			# Emit UpdateAgentTaskEvent at the END of run() with final task state
			self.eventbus.dispatch(UpdateAgentTaskEvent.from_agent(self))
			await self._cache_qa_login_state()

			# Generate GIF if needed before stopping event bus
			if self.settings.generate_gif:
				output_path: str = 'agent_history.gif'
				if isinstance(self.settings.generate_gif, str):
					output_path = self.settings.generate_gif

				# Lazy import gif module to avoid heavy startup cost
				from browser_use.agent.gif import create_history_gif

				create_history_gif(task=self.task, history=self.history, output_path=output_path)

				# Only emit output file event if GIF was actually created
				if Path(output_path).exists():
					output_event = await CreateAgentOutputFileEvent.from_agent_and_file(self, output_path)
					self.eventbus.dispatch(output_event)

			# Log final messages to user based on outcome
			self._log_final_outcome_messages()

			# Stop the event bus gracefully, waiting for all events to be processed
			# Configurable via TIMEOUT_AgentEventBusStop env var (default: 3.0s)
			await self.eventbus.stop(clear=True, timeout=_get_timeout('TIMEOUT_AgentEventBusStop', 3.0))

			await self.close()

	@observe_debug(ignore_input=True, ignore_output=True)
	@time_execution_async('--multi_act')
	async def multi_act(self, actions: list[ActionModel]) -> list[ActionResult]:
		"""Execute multiple actions with page-change guards.

		Two layers of protection prevent executing actions against stale DOM:
		  1. Static flag: actions tagged with terminates_sequence=True (navigate, search, go_back, switch)
		     automatically abort remaining queued actions.
		  2. Runtime detection: after every action, the current URL and focused target are compared
		     to pre-action values. Any change aborts the remaining queue.
		"""
		results: list[ActionResult] = []
		total_actions = len(actions)

		assert self.browser_session is not None, 'BrowserSession is not set up'
		try:
			if (
				self.browser_session._cached_browser_state_summary is not None
				and self.browser_session._cached_browser_state_summary.dom_state is not None
			):
				cached_selector_map = dict(self.browser_session._cached_browser_state_summary.dom_state.selector_map)
			else:
				cached_selector_map = {}
		except Exception as e:
			self.logger.error(f'Error getting cached selector map: {e}')
			cached_selector_map = {}

		for i, action in enumerate(actions):
			# Get action name from the action model BEFORE try block to ensure it's always available in except
			action_data = action.model_dump(exclude_unset=True)
			action_name = next(iter(action_data.keys())) if action_data else 'unknown'

			if i > 0:
				# The QA completion action must always be emitted alone.
				if action_data.get(self._completion_action_name) is not None:
					msg = f'finish_test_step is allowed only as a single action - stopped after action {i} / {total_actions}.'
					self.logger.debug(msg)
					break

			# wait between actions (only after first action)
			if i > 0:
				self.logger.debug(f'Waiting {self.browser_profile.wait_between_actions} seconds between actions')
				await asyncio.sleep(self.browser_profile.wait_between_actions)

			try:
				await self._check_stop_or_pause()

				# Log action before execution
				await self._log_action(action, action_name, i + 1, total_actions)

				# Capture pre-action state for runtime page-change detection
				pre_action_url = await self.browser_session.get_current_page_url()
				pre_action_focus = self.browser_session.agent_focus_target_id

				result = await self.tools.act(
					action=action,
					browser_session=self.browser_session,
					file_system=self.file_system,
					page_extraction_llm=self.settings.page_extraction_llm,
					sensitive_data=self.sensitive_data,
					available_file_paths=self.available_file_paths,
					extraction_schema=self.extraction_schema,
				)

				if result.error:
					await self._demo_mode_log(
						f'Action "{action_name}" failed: {result.error}',
						'error',
						{'action': action_name, 'step': self.state.n_steps},
					)
				elif result.is_done:
					completion_text = result.long_term_memory or result.extracted_content or 'Task marked as done.'
					level = 'success' if result.success is not False else 'warning'
					await self._demo_mode_log(
						completion_text,
						level,
						{'action': action_name, 'step': self.state.n_steps},
					)

				results.append(result)

				if results[-1].is_done or results[-1].error or i == total_actions - 1:
					break

				# --- Page-change guards (only when more actions remain) ---

				# Layer 1: Static flag — action metadata declares it changes the page
				registered_action = self.tools.registry.registry.actions.get(action_name)
				if registered_action and registered_action.terminates_sequence:
					self.logger.info(
						f'Action "{action_name}" terminates sequence — skipping {total_actions - i - 1} remaining action(s)'
					)
					break

				# Layer 2: Runtime detection — URL or focus target changed
				post_action_url = await self.browser_session.get_current_page_url()
				post_action_focus = self.browser_session.agent_focus_target_id

				if post_action_url != pre_action_url or post_action_focus != pre_action_focus:
					self.logger.info(f'Page changed after "{action_name}" — skipping {total_actions - i - 1} remaining action(s)')
					break

			except Exception as e:
				# Re-raise InterruptedError so _check_stop_or_pause's stop/pause signal still propagates
				if isinstance(e, InterruptedError):
					raise
				# Re-raise browser/connection errors so _handle_step_error can handle reconnect/shutdown
				if self._is_connection_like_error(e):
					raise
				# Handle any exceptions during action execution
				self.logger.error(f'❌ Executing action {i + 1} failed -> {type(e).__name__}: {e}')
				await self._demo_mode_log(
					f'Action "{action_name}" raised {type(e).__name__}: {e}',
					'error',
					{'action': action_name, 'step': self.state.n_steps},
				)
				# Preserve partial results so the agent knows which actions succeeded before the failure
				results.append(ActionResult(error=f'{type(e).__name__}: {e}'))
				return results

		return results

	async def _log_action(self, action, action_name: str, action_num: int, total_actions: int) -> None:
		"""Log the action before execution with colored formatting"""
		# Color definitions
		blue = '\033[34m'  # Action name
		magenta = '\033[35m'  # Parameter names
		reset = '\033[0m'

		# Format action number and name
		if total_actions > 1:
			action_header = f'▶️  [{action_num}/{total_actions}] {blue}{action_name}{reset}:'
			plain_header = f'▶️  [{action_num}/{total_actions}] {action_name}:'
		else:
			action_header = f'▶️   {blue}{action_name}{reset}:'
			plain_header = f'▶️  {action_name}:'

		# Get action parameters
		action_data = action.model_dump(exclude_unset=True)
		params = action_data.get(action_name, {})

		# Build parameter parts with colored formatting
		param_parts = []
		plain_param_parts = []

		if params and isinstance(params, dict):
			for param_name, value in params.items():
				# Truncate long values for readability
				if isinstance(value, str) and len(value) > 150:
					display_value = value[:150] + '...'
				elif isinstance(value, list) and len(str(value)) > 200:
					display_value = str(value)[:200] + '...'
				else:
					display_value = value

				param_parts.append(f'{magenta}{param_name}{reset}: {display_value}')
				plain_param_parts.append(f'{param_name}: {display_value}')

		# Join all parts
		if param_parts:
			params_string = ', '.join(param_parts)
			self.logger.info(f'  {action_header} {params_string}')
		else:
			self.logger.info(f'  {action_header}')

		if self._demo_mode_enabled:
			panel_message = plain_header
			if plain_param_parts:
				panel_message = f'{panel_message} {", ".join(plain_param_parts)}'
			await self._demo_mode_log(panel_message.strip(), 'action', {'action': action_name, 'step': self.state.n_steps})

	async def log_completion(self) -> None:
		"""Log the completion of the task"""
		# self._task_end_time = time.time()
		# self._task_duration = self._task_end_time - self._task_start_time TODO: this is not working when using take_step
		if self.history.is_successful():
			self.logger.info('✅ Task completed successfully')
			await self._demo_mode_log('Task completed successfully', 'success', {'tag': 'task'})

	async def _generate_rerun_summary(
		self, original_task: str, results: list[ActionResult], summary_llm: BaseChatModel | None = None
	) -> ActionResult:
		"""Generate AI summary of rerun completion using screenshot and last step info"""
		from browser_use.agent.views import RerunSummaryAction

		# Get current screenshot
		screenshot_b64 = None
		try:
			screenshot = await self.browser_session.take_screenshot(full_page=False)
			if screenshot:
				import base64

				screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
		except Exception as e:
			self.logger.warning(f'Failed to capture screenshot for rerun summary: {e}')

		# Build summary prompt and message
		error_count = sum(1 for r in results if r.error)
		success_count = len(results) - error_count

		from browser_use.agent.prompts import get_rerun_summary_message, get_rerun_summary_prompt

		prompt = get_rerun_summary_prompt(
			original_task=original_task,
			total_steps=len(results),
			success_count=success_count,
			error_count=error_count,
		)

		# Use provided LLM, agent's LLM, or fall back to OpenAI with structured output
		try:
			# Determine which LLM to use
			if summary_llm is None:
				# Try to use the agent's LLM first
				summary_llm = self.llm
				self.logger.debug('Using agent LLM for rerun summary')
			else:
				self.logger.debug(f'Using provided LLM for rerun summary: {summary_llm.model}')

			# Build message with prompt and optional screenshot
			from browser_use.llm.messages import BaseMessage

			message = get_rerun_summary_message(prompt, screenshot_b64)
			messages: list[BaseMessage] = [message]  # type: ignore[list-item]

			# Try calling with structured output first
			self.logger.debug(f'Calling LLM for rerun summary with {len(messages)} message(s)')
			try:
				kwargs: dict = {'output_format': RerunSummaryAction}
				response = await summary_llm.ainvoke(messages, **kwargs)
				summary: RerunSummaryAction = response.completion  # type: ignore[assignment]
				self.logger.debug(f'LLM response type: {type(summary)}')
				self.logger.debug(f'LLM response: {summary}')
			except Exception as structured_error:
				# If structured output fails (e.g., Browser-Use LLM doesn't support it for this type),
				# fall back to text response without parsing
				self.logger.debug(f'Structured output failed: {structured_error}, falling back to text response')

				response = await summary_llm.ainvoke(messages, None)
				response_text = response.completion
				self.logger.debug(f'LLM text response: {response_text}')

				# Use the text response directly as the summary
				summary = RerunSummaryAction(
					summary=response_text if isinstance(response_text, str) else str(response_text),
					success=error_count == 0,
					completion_status='complete' if error_count == 0 else ('partial' if success_count > 0 else 'failed'),
				)

			self.logger.info(f'📊 Rerun Summary: {summary.summary}')
			self.logger.info(f'📊 Status: {summary.completion_status} (success={summary.success})')

			return ActionResult(
				is_done=True,
				success=summary.success,
				extracted_content=summary.summary,
				long_term_memory=f'Rerun completed with status: {summary.completion_status}. {summary.summary[:100]}',
			)

		except Exception as e:
			self.logger.warning(f'Failed to generate AI summary: {e.__class__.__name__}: {e}')
			self.logger.debug('Full error traceback:', exc_info=True)
			# Fallback to simple summary
			return ActionResult(
				is_done=True,
				success=error_count == 0,
				extracted_content=f'Rerun completed: {success_count}/{len(results)} steps succeeded',
				long_term_memory=f'Rerun completed: {success_count} steps succeeded, {error_count} errors',
			)

	async def _execute_ai_step(
		self,
		query: str,
		include_screenshot: bool = False,
		extract_links: bool = False,
		ai_step_llm: BaseChatModel | None = None,
	) -> ActionResult:
		"""
		Execute an AI step during rerun to re-evaluate extract actions.
		Analyzes full page DOM/markdown + optional screenshot.

		Args:
			query: What to analyze or extract from the current page
			include_screenshot: Whether to include screenshot in analysis
			extract_links: Whether to include links in markdown extraction
			ai_step_llm: Optional LLM to use. If not provided, uses agent's LLM

		Returns:
			ActionResult with extracted content
		"""
		from browser_use.agent.prompts import get_ai_step_system_prompt, get_ai_step_user_prompt, get_rerun_summary_message
		from browser_use.llm.messages import SystemMessage, UserMessage
		from browser_use.utils import sanitize_surrogates

		# Use provided LLM or agent's LLM
		llm = ai_step_llm or self.llm
		self.logger.debug(f'Using LLM for AI step: {llm.model}')

		# Extract clean markdown
		try:
			from browser_use.dom.markdown_extractor import extract_clean_markdown

			content, content_stats = await extract_clean_markdown(
				browser_session=self.browser_session, extract_links=extract_links
			)
		except Exception as e:
			return ActionResult(error=f'Could not extract clean markdown: {type(e).__name__}: {e}')

		# Get screenshot if requested
		screenshot_b64 = None
		if include_screenshot:
			try:
				screenshot = await self.browser_session.take_screenshot(full_page=False)
				if screenshot:
					import base64

					screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
			except Exception as e:
				self.logger.warning(f'Failed to capture screenshot for ai_step: {e}')

		# Build prompt with content stats
		original_html_length = content_stats['original_html_chars']
		initial_markdown_length = content_stats['initial_markdown_chars']
		final_filtered_length = content_stats['final_filtered_chars']
		chars_filtered = content_stats['filtered_chars_removed']

		stats_summary = f"""Content processed: {original_html_length:,} HTML chars → {initial_markdown_length:,} initial markdown → {final_filtered_length:,} filtered markdown"""
		if chars_filtered > 0:
			stats_summary += f' (filtered {chars_filtered:,} chars of noise)'

		# Sanitize content
		content = sanitize_surrogates(content)
		query = sanitize_surrogates(query)

		# Get prompts from prompts.py
		system_prompt = get_ai_step_system_prompt()
		prompt_text = get_ai_step_user_prompt(query, stats_summary, content)

		# Build user message with optional screenshot
		if screenshot_b64:
			user_message = get_rerun_summary_message(prompt_text, screenshot_b64)
		else:
			user_message = UserMessage(content=prompt_text)

		try:
			import asyncio

			response = await asyncio.wait_for(llm.ainvoke([SystemMessage(content=system_prompt), user_message]), timeout=120.0)

			current_url = await self.browser_session.get_current_page_url()
			extracted_content = (
				f'<url>\n{current_url}\n</url>\n<query>\n{query}\n</query>\n<result>\n{response.completion}\n</result>'
			)

			# Simple memory handling
			MAX_MEMORY_LENGTH = 1000
			if len(extracted_content) < MAX_MEMORY_LENGTH:
				memory = extracted_content
				include_extracted_content_only_once = False
			else:
				file_name = await self.file_system.save_extracted_content(extracted_content)
				memory = f'Query: {query}\nContent in {file_name} and once in <read_state>.'
				include_extracted_content_only_once = True

			self.logger.info(f'🤖 AI Step: {memory}')
			return ActionResult(
				extracted_content=extracted_content,
				include_extracted_content_only_once=include_extracted_content_only_once,
				long_term_memory=memory,
			)
		except Exception as e:
			self.logger.warning(f'Failed to execute AI step: {e.__class__.__name__}: {e}')
			self.logger.debug('Full error traceback:', exc_info=True)
			return ActionResult(error=f'AI step failed: {e}')

	async def rerun_history(
		self,
		history: AgentHistoryList,
		max_retries: int = 3,
		skip_failures: bool = False,
		delay_between_actions: float = 2.0,
		max_step_interval: float = 45.0,
		summary_llm: BaseChatModel | None = None,
		ai_step_llm: BaseChatModel | None = None,
		wait_for_elements: bool = False,
	) -> list[ActionResult]:
		"""
		Rerun a saved history of actions with error handling and retry logic.

		Args:
		                history: The history to replay
		                max_retries: Maximum number of retries per action
		                skip_failures: Whether to skip failed actions or stop execution. When True, also skips
		                               steps that had errors in the original run (e.g., modal close buttons that
		                               auto-dismissed, or elements that became non-interactable)
		                delay_between_actions: Delay between actions in seconds (used when no saved interval)
		                max_step_interval: Maximum delay from saved step_interval (caps LLM time from original run)
		                summary_llm: Optional LLM to use for generating the final summary. If not provided, uses the agent's LLM
		                ai_step_llm: Optional LLM to use for AI steps (extract actions). If not provided, uses the agent's LLM
		                wait_for_elements: If True, wait until the saved target elements are matchable before
		                               replaying the step. Useful for SPA pages where shadow DOM content loads
		                               dynamically. Default is False.

		Returns:
		                List of action results (including AI summary as the final result)
		"""
		# Skip cloud sync session events for rerunning (we're replaying, not starting new)
		self.state.session_initialized = True

		# Initialize browser session
		await self.browser_session.start()

		results = []

		# Track previous step for redundant retry detection
		previous_item: AgentHistory | None = None
		previous_step_succeeded: bool = False

		try:
			for i, history_item in enumerate(history.history):
				goal = history_item.model_output.current_state.next_goal if history_item.model_output else ''
				step_num = history_item.metadata.step_number if history_item.metadata else i
				step_name = 'Initial actions' if step_num == 0 else f'Step {step_num}'

				# Determine step delay
				if history_item.metadata and history_item.metadata.step_interval is not None:
					# Cap the saved interval to max_step_interval (saved interval includes LLM time)
					step_delay = min(history_item.metadata.step_interval, max_step_interval)
					# Format delay nicely - show ms for values < 1s, otherwise show seconds
					if step_delay < 1.0:
						delay_str = f'{step_delay * 1000:.0f}ms'
					else:
						delay_str = f'{step_delay:.1f}s'
					if history_item.metadata.step_interval > max_step_interval:
						delay_source = f'capped to {delay_str} (saved was {history_item.metadata.step_interval:.1f}s)'
					else:
						delay_source = f'using saved step_interval={delay_str}'
				else:
					step_delay = delay_between_actions
					if step_delay < 1.0:
						delay_str = f'{step_delay * 1000:.0f}ms'
					else:
						delay_str = f'{step_delay:.1f}s'
					delay_source = f'using default delay={delay_str}'

				self.logger.info(f'Replaying {step_name} ({i + 1}/{len(history.history)}) [{delay_source}]: {goal}')

				if (
					not history_item.model_output
					or not history_item.model_output.action
					or history_item.model_output.action == [None]
				):
					self.logger.warning(f'{step_name}: No action to replay, skipping')
					results.append(ActionResult(error='No action to replay'))
					continue

				# Check if the original step had errors - skip if skip_failures is enabled
				original_had_error = any(r.error for r in history_item.result if r.error)
				if original_had_error and skip_failures:
					error_msgs = [r.error for r in history_item.result if r.error]
					self.logger.warning(
						f'{step_name}: Original step had error(s), skipping (skip_failures=True): {error_msgs[0][:100] if error_msgs else "unknown"}'
					)
					results.append(
						ActionResult(
							error=f'Skipped - original step had error: {error_msgs[0][:100] if error_msgs else "unknown"}'
						)
					)
					continue

				# Check if this step is a redundant retry of the previous step
				# This handles cases where original run needed to click same element multiple times
				# due to slow page response, but during replay the first click already worked
				if self._is_redundant_retry_step(history_item, previous_item, previous_step_succeeded):
					self.logger.info(f'{step_name}: Skipping redundant retry (previous step already succeeded with same element)')
					results.append(
						ActionResult(
							extracted_content='Skipped - redundant retry of previous step',
							include_in_memory=False,
						)
					)
					# Don't update previous_item/previous_step_succeeded - keep tracking the original step
					continue

				retry_count = 0
				step_succeeded = False
				menu_reopened = False  # Track if we've already tried reopening the menu
				# Exponential backoff: 5s base, doubling each retry, capped at 30s
				base_retry_delay = 5.0
				max_retry_delay = 30.0
				while retry_count < max_retries:
					try:
						result = await self._execute_history_step(history_item, step_delay, ai_step_llm, wait_for_elements)
						results.extend(result)
						step_succeeded = True
						break

					except Exception as e:
						error_str = str(e)
						retry_count += 1

						# Check if this is a "Could not find matching element" error for a menu item
						# If so, try to re-open the dropdown from the previous step before retrying
						if (
							not menu_reopened
							and 'Could not find matching element' in error_str
							and previous_item is not None
							and self._is_menu_opener_step(previous_item)
						):
							# Check if current step targets a menu item element
							curr_elements = history_item.state.interacted_element if history_item.state else []
							curr_elem = curr_elements[0] if curr_elements else None
							if self._is_menu_item_element(curr_elem):
								self.logger.info(
									'🔄 Dropdown may have closed. Attempting to re-open by re-executing previous step...'
								)
								reopened = await self._reexecute_menu_opener(previous_item, ai_step_llm)
								if reopened:
									menu_reopened = True
									# Don't increment retry_count for the menu reopen attempt
									# Retry immediately with minimal delay
									retry_count -= 1
									step_delay = 0.5  # Use short delay after reopening
									self.logger.info('🔄 Dropdown re-opened, retrying element match...')
									continue

						if retry_count == max_retries:
							error_msg = f'{step_name} failed after {max_retries} attempts: {error_str}'
							self.logger.error(error_msg)
							# Always record the error in results so AI summary counts it correctly
							results.append(ActionResult(error=error_msg))
							if not skip_failures:
								raise RuntimeError(error_msg)
							# With skip_failures=True, continue to next step
						else:
							# Exponential backoff: 5s, 10s, 20s, ... capped at 30s
							retry_delay = min(base_retry_delay * (2 ** (retry_count - 1)), max_retry_delay)
							self.logger.warning(
								f'{step_name} failed (attempt {retry_count}/{max_retries}), retrying in {retry_delay}s...'
							)
							await asyncio.sleep(retry_delay)

				# Update tracking for redundant retry detection
				previous_item = history_item
				previous_step_succeeded = step_succeeded

			# Generate AI summary of rerun completion
			self.logger.info('🤖 Generating AI summary of rerun completion...')
			summary_result = await self._generate_rerun_summary(self.task, results, summary_llm)
			results.append(summary_result)

			return results
		finally:
			# Always close resources, even on failure
			await self.close()

	async def _execute_initial_actions(self) -> None:
		# Execute initial actions if provided
		if self.initial_actions and not self.state.follow_up_task:
			self.logger.debug(f'⚡ Executing {len(self.initial_actions)} initial actions...')
			result = await self.multi_act(self.initial_actions)
			# update result 1 to mention that its was automatically loaded
			if result and self.initial_url and result[0].long_term_memory:
				result[0].long_term_memory = f'Found initial url and automatically loaded it. {result[0].long_term_memory}'
			self.state.last_result = result

			# Save initial actions to history as step 0 for rerun capability
			# Skip browser state capture for initial actions (usually just URL navigation)
			if self.settings.flash_mode:
				model_output = self.AgentOutput(
					evaluation_previous_goal=None,
					memory='Initial navigation',
					next_goal=None,
					action=self.initial_actions,
				)
			else:
				model_output = self.AgentOutput(
					evaluation_previous_goal='Start',
					memory=None,
					next_goal='Initial navigation',
					action=self.initial_actions,
				)

			metadata = StepMetadata(step_number=0, step_start_time=time.time(), step_end_time=time.time(), step_interval=None)

			# Create minimal browser state history for initial actions
			state_history = BrowserStateHistory(
				url=self.initial_url or '',
				title='Initial Actions',
				tabs=[],
				interacted_element=[None] * len(self.initial_actions),  # No DOM elements needed
				screenshot_path=None,
			)

			history_item = AgentHistory(
				model_output=model_output,
				result=result,
				state=state_history,
				metadata=metadata,
			)

			self.history.add_item(history_item)
			self.logger.debug('📝 Saved initial actions to history as step 0')
			self.logger.debug('Initial actions completed')

	@staticmethod
	def _historical_element_is_present(
		historical_element: DOMInteractedElement,
		browser_state_summary: BrowserStateSummary,
	) -> bool:
		"""Return whether a saved element can be matched in the current DOM state."""

		selector_map = browser_state_summary.dom_state.selector_map or {}
		selector_items = list(selector_map.items())
		if historical_element.frame_id:
			same_frame_items = [
				(index, element) for index, element in selector_items if element.frame_id == historical_element.frame_id
			]
			if same_frame_items:
				selector_items = same_frame_items + [
					(index, element) for index, element in selector_items if element.frame_id != historical_element.frame_id
				]

		if any(element.element_hash == historical_element.element_hash for _, element in selector_items):
			return True
		if historical_element.stable_hash is not None and any(
			element.compute_stable_hash() == historical_element.stable_hash for _, element in selector_items
		):
			return True
		if historical_element.x_path and any(element.xpath == historical_element.x_path for _, element in selector_items):
			return True
		if historical_element.ax_name:
			node_name = historical_element.node_name.lower()
			if any(
				element.node_name.lower() == node_name
				and element.ax_node is not None
				and element.ax_node.name == historical_element.ax_name
				for _, element in selector_items
			):
				return True
		if historical_element.attributes:
			node_name = historical_element.node_name.lower()
			for attribute in ('name', 'id', 'aria-label'):
				value = historical_element.attributes.get(attribute)
				if value and any(
					element.node_name.lower() == node_name and element.attributes and element.attributes.get(attribute) == value
					for _, element in selector_items
				):
					return True
		return False

	def _history_targets_present(
		self,
		history_item: AgentHistory,
		browser_state_summary: BrowserStateSummary,
	) -> bool:
		"""Check every saved interactive target instead of relying on old indices."""

		if history_item.model_output is None:
			return True
		for index, action in enumerate(history_item.model_output.action):
			action_name = self._qa_replay_action_name(action)
			if action_name not in {'click', 'input', 'hover', 'select_option', 'drag_and_drop'}:
				continue
			historical_element = (
				history_item.state.interacted_element[index] if index < len(history_item.state.interacted_element) else None
			)
			if historical_element is not None and not self._historical_element_is_present(
				historical_element, browser_state_summary
			):
				return False
		return True

	@staticmethod
	def _qa_replay_page_key(url: str) -> tuple[str, str, int | None, str] | None:
		"""Return a conservative page identity for comparing saved and current states."""

		if not url:
			return None
		parsed = urlparse(url)
		if not parsed.scheme or not parsed.hostname:
			return None
		path = parsed.path.rstrip('/') or '/'
		return parsed.scheme.lower(), parsed.hostname.lower(), parsed.port, path

	def _qa_replay_authentication_batch_is_obsolete(
		self,
		history_item: AgentHistory,
		browser_state_summary: BrowserStateSummary,
	) -> bool:
		"""Return whether restored login state has already completed this saved auth batch."""

		if not self.settings.reuse_login_state or not self._qa_login_storage_state:
			return False
		historical_page = self._qa_replay_page_key(history_item.state.url)
		current_page = self._qa_replay_page_key(browser_state_summary.url)
		if historical_page is None or current_page is None or historical_page == current_page:
			return False

		historical_path = historical_page[3].lower()
		if any(marker in historical_path for marker in ('/login', '/sign-in', '/signin', '/authentication')):
			return True

		for element in history_item.state.interacted_element:
			attributes = element.attributes if element is not None and element.attributes else {}
			field_type = attributes.get('type', '').lower()
			autocomplete = attributes.get('autocomplete', '').lower()
			if field_type == 'password' or autocomplete in {'username', 'current-password'}:
				return True
		return False

	async def _wait_for_history_targets(
		self,
		history_item: AgentHistory,
		*,
		timeout: float = 15.0,
		poll_interval: float = 0.5,
		initial_state: BrowserStateSummary | None = None,
	) -> BrowserStateSummary:
		"""Wait only until the saved target elements are objectively matchable."""

		started_at = time.perf_counter()
		last_state = initial_state
		while True:
			if last_state is None:
				last_state = await self.browser_session.get_browser_state_summary(include_screenshot=False)
			if self._history_targets_present(history_item, last_state):
				return last_state
			remaining = timeout - (time.perf_counter() - started_at)
			if remaining <= 0:
				break
			await asyncio.sleep(min(poll_interval, remaining))
			last_state = None

		self.logger.warning(f'⚠️ Timeout waiting {timeout:.1f}s for saved replay targets; attempting final match')
		assert last_state is not None
		return last_state

	async def _wait_for_minimum_elements(
		self,
		min_elements: int,
		timeout: float = 30.0,
		poll_interval: float = 1.0,
	) -> BrowserStateSummary | None:
		"""Wait for the page to have at least min_elements interactive elements.

		This helps handle SPA pages where shadow DOM and dynamic content
		may not be immediately available even when document.readyState is 'complete'.

		Args:
			min_elements: Minimum number of interactive elements to wait for
			timeout: Maximum time to wait in seconds
			poll_interval: Time between polling attempts in seconds

		Returns:
			BrowserStateSummary if minimum elements found, None if timeout
		"""
		assert self.browser_session is not None, 'BrowserSession is not set up'

		start_time = time.time()
		last_count = 0

		while (time.time() - start_time) < timeout:
			state = await self.browser_session.get_browser_state_summary(include_screenshot=False)
			if state and state.dom_state.selector_map:
				current_count = len(state.dom_state.selector_map)
				if current_count >= min_elements:
					self.logger.debug(f'✅ Page has {current_count} elements (needed {min_elements}), proceeding with action')
					return state
				if current_count != last_count:
					self.logger.debug(
						f'⏳ Waiting for elements: {current_count}/{min_elements} '
						f'(timeout in {timeout - (time.time() - start_time):.1f}s)'
					)
					last_count = current_count
			await asyncio.sleep(poll_interval)

		# Return last state even if we didn't reach min_elements
		self.logger.warning(f'⚠️ Timeout waiting for {min_elements} elements, proceeding with {last_count} elements')
		return await self.browser_session.get_browser_state_summary(include_screenshot=False)

	def _count_expected_elements_from_history(self, history_item: AgentHistory) -> int:
		"""Estimate the minimum number of elements expected based on history.

		Uses the action indices from the history to determine the minimum
		number of elements the page should have. If an action targets index N,
		the page needs at least N+1 elements in the selector_map.
		"""
		if not history_item.model_output or not history_item.model_output.action:
			return 0

		max_index = -1  # Use -1 to indicate no index found yet
		for action in history_item.model_output.action:
			# Get the element index this action targets
			index = action.get_index()
			if index is not None:
				max_index = max(max_index, index)

		# Need at least max_index + 1 elements (indices are 0-based)
		# Cap at 50 to avoid waiting forever for very high indices
		# max_index >= 0 means we found at least one action with an index
		return min(max_index + 1, 50) if max_index >= 0 else 0

	async def _execute_history_step(
		self,
		history_item: AgentHistory,
		delay: float,
		ai_step_llm: BaseChatModel | None = None,
		wait_for_elements: bool = False,
		browser_state_summary: BrowserStateSummary | None = None,
	) -> list[ActionResult]:
		"""Execute a single step from history with element validation.

		For extract actions, uses AI to re-evaluate the content since page content may have changed.

		Args:
			history_item: The history step to execute
			delay: Delay before executing the step
			ai_step_llm: Optional LLM to use for AI steps
			wait_for_elements: If True, wait until saved target elements are matchable
			browser_state_summary: Optional already-captured state to reuse for element matching
		"""
		assert self.browser_session is not None, 'BrowserSession is not set up'

		await asyncio.sleep(delay)

		if browser_state_summary is not None:
			state = browser_state_summary
		# Optionally wait for the saved target instead of an unrelated element count.
		elif wait_for_elements:
			# Determine if we need to wait for elements (actions that interact with DOM elements)
			needs_element_matching = False
			if history_item.model_output:
				for i, action in enumerate(history_item.model_output.action):
					action_data = action.model_dump(exclude_unset=True)
					action_name = next(iter(action_data.keys()), None)
					# Actions that need element matching
					if action_name in ('click', 'input', 'hover', 'select_option', 'drag_and_drop'):
						historical_elem = (
							history_item.state.interacted_element[i] if i < len(history_item.state.interacted_element) else None
						)
						if historical_elem is not None:
							needs_element_matching = True
							break

			# If we need element matching, proceed as soon as every saved target is matchable.
			if needs_element_matching:
				state = await self._wait_for_history_targets(history_item, timeout=15.0, poll_interval=0.5)
			else:
				state = await self.browser_session.get_browser_state_summary(include_screenshot=False)
		else:
			state = await self.browser_session.get_browser_state_summary(include_screenshot=False)
		if not state or not history_item.model_output:
			raise ValueError('Invalid state or model output')

		results = []
		pending_actions = []

		for i, action in enumerate(history_item.model_output.action):
			# Check if this is an extract action - use AI step instead
			action_data = action.model_dump(exclude_unset=True)
			action_name = next(iter(action_data.keys()), None)

			if action_name == 'extract':
				# Execute any pending actions first to maintain correct order
				# (e.g., if step is [click, extract], click must happen before extract)
				if pending_actions:
					batch_results = await self.multi_act(pending_actions)
					results.extend(batch_results)
					pending_actions = []

				# Now execute AI step for extract action
				extract_params = action_data['extract']
				query = extract_params.get('query', '')
				extract_links = extract_params.get('extract_links', False)

				self.logger.info(f'🤖 Using AI step for extract action: {query[:50]}...')
				ai_result = await self._execute_ai_step(
					query=query,
					include_screenshot=False,  # Match original extract behavior
					extract_links=extract_links,
					ai_step_llm=ai_step_llm,
				)
				results.append(ai_result)
			else:
				# For non-extract actions, update indices and collect for batch execution
				historical_elem = history_item.state.interacted_element[i]
				updated_action = await self._update_action_indices(
					historical_elem,
					action,
					state,
				)
				if updated_action is None:
					# Build informative error message with diagnostic info
					elem_info = self._format_element_for_error(historical_elem)
					selector_map = state.dom_state.selector_map or {}
					selector_count = len(selector_map)

					# Find elements with same node_name for diagnostics
					hist_node = historical_elem.node_name.lower() if historical_elem else ''
					similar_elements = []
					if historical_elem and historical_elem.attributes:
						for idx, elem in selector_map.items():
							if elem.node_name.lower() == hist_node and elem.attributes:
								elem_aria = elem.attributes.get('aria-label', '')
								if elem_aria:
									similar_elements.append(f'{idx}:{elem_aria[:30]}')
									if len(similar_elements) >= 5:
										break

					diagnostic = ''
					if similar_elements:
						diagnostic = f'\n  Available <{hist_node.upper()}> with aria-label: {similar_elements}'
					elif hist_node:
						same_node_count = sum(1 for e in selector_map.values() if e.node_name.lower() == hist_node)
						diagnostic = (
							f'\n  Found {same_node_count} <{hist_node.upper()}> elements (none with matching identifiers)'
						)

					raise ValueError(
						f'Could not find matching element for action {i} in current page.\n'
						f'  Looking for: {elem_info}\n'
						f'  Page has {selector_count} interactive elements.{diagnostic}\n'
						f'  Tried: EXACT hash → STABLE hash → XPATH → AX_NAME → ATTRIBUTE matching'
					)
				pending_actions.append(updated_action)

		# Execute any remaining pending actions
		if pending_actions:
			batch_results = await self.multi_act(pending_actions)
			results.extend(batch_results)

		return results

	async def _update_action_indices(
		self,
		historical_element: DOMInteractedElement | None,
		action: ActionModel,  # Type this properly based on your action model
		browser_state_summary: BrowserStateSummary,
	) -> ActionModel | None:
		"""
		Update action indices based on current page state.
		Returns updated action or None if element cannot be found.

		Cascading matching strategy (tries each level in order):
		1. EXACT: Full element_hash match (includes all attributes + ax_name)
		2. STABLE: Hash with dynamic CSS classes filtered out (focus, hover, animation, etc.)
		3. XPATH: XPath string match (structural position in DOM)
		4. AX_NAME: Accessible name match from accessibility tree (robust for dynamic menus)
		5. ATTRIBUTE: Unique attribute match (name, id, aria-label) for old history files
		"""
		if not historical_element or not browser_state_summary.dom_state.selector_map:
			return action

		selector_map = browser_state_summary.dom_state.selector_map
		selector_items = list(selector_map.items())
		if historical_element.frame_id:
			same_frame_items = [
				(index, element) for index, element in selector_items if element.frame_id == historical_element.frame_id
			]
			if same_frame_items:
				selector_items = same_frame_items + [
					(index, element) for index, element in selector_items if element.frame_id != historical_element.frame_id
				]
		highlight_index: int | None = None
		match_level: MatchLevel | None = None

		# Debug: log what we're looking for and what's available
		self.logger.info(
			f'🔍 Searching for element: <{historical_element.node_name}> '
			f'hash={historical_element.element_hash} stable_hash={historical_element.stable_hash}'
		)
		# Log what elements are in selector_map for debugging
		if historical_element.node_name:
			hist_name = historical_element.node_name.lower()
			matching_nodes = [
				(idx, elem.node_name, elem.attributes.get('name') if elem.attributes else None)
				for idx, elem in selector_items
				if elem.node_name.lower() == hist_name
			]
			self.logger.info(
				f'🔍 Selector map has {len(selector_map)} elements, '
				f'{len(matching_nodes)} are <{hist_name.upper()}>: {matching_nodes}'
			)

		# Level 1: EXACT hash match
		for idx, elem in selector_items:
			if elem.element_hash == historical_element.element_hash:
				highlight_index = idx
				match_level = MatchLevel.EXACT
				break

		if highlight_index is None:
			self.logger.debug(f'EXACT hash match failed (checked {len(selector_map)} elements)')

		# Level 2: STABLE hash match (dynamic classes filtered)
		# Use stored stable_hash (computed at save time from EnhancedDOMTreeNode - single source of truth)
		if highlight_index is None and historical_element.stable_hash is not None:
			for idx, elem in selector_items:
				if elem.compute_stable_hash() == historical_element.stable_hash:
					highlight_index = idx
					match_level = MatchLevel.STABLE
					self.logger.info('Element matched at STABLE level (dynamic classes filtered)')
					break
			if highlight_index is None:
				self.logger.debug('STABLE hash match failed')
		elif highlight_index is None:
			self.logger.debug('STABLE hash match skipped (no stable_hash in history)')

		# Level 3: XPATH match
		if highlight_index is None and historical_element.x_path:
			for idx, elem in selector_items:
				if elem.xpath == historical_element.x_path:
					highlight_index = idx
					match_level = MatchLevel.XPATH
					self.logger.info(f'Element matched at XPATH level: {historical_element.x_path}')
					break
			if highlight_index is None:
				self.logger.debug(f'XPATH match failed for: {historical_element.x_path[-60:]}')

		# Level 4: ax_name (accessible name) match - robust for dynamic SPAs with menus
		# This uses the accessible name from the accessibility tree which is stable
		# even when DOM structure changes (e.g., dynamically generated menu items)
		if highlight_index is None and historical_element.ax_name:
			hist_name = historical_element.node_name.lower()
			hist_ax_name = historical_element.ax_name
			for idx, elem in selector_items:
				# Match by node type and accessible name
				elem_ax_name = elem.ax_node.name if elem.ax_node else None
				if elem.node_name.lower() == hist_name and elem_ax_name == hist_ax_name:
					highlight_index = idx
					match_level = MatchLevel.AX_NAME
					self.logger.info(f'Element matched at AX_NAME level: "{hist_ax_name}"')
					break
			if highlight_index is None:
				# Log available ax_names for debugging
				same_type_ax_names = [
					(idx, elem.ax_node.name if elem.ax_node else None)
					for idx, elem in selector_items
					if elem.node_name.lower() == hist_name and elem.ax_node and elem.ax_node.name
				]
				self.logger.debug(
					f'AX_NAME match failed for <{hist_name.upper()}> ax_name="{hist_ax_name}". '
					f'Page has {len(same_type_ax_names)} <{hist_name.upper()}> with ax_names: '
					f'{same_type_ax_names[:5]}{"..." if len(same_type_ax_names) > 5 else ""}'
				)

		# Level 5: Unique attribute fallback (for old history files without stable_hash)
		if highlight_index is None and historical_element.attributes:
			hist_attrs = historical_element.attributes
			hist_name = historical_element.node_name.lower()

			# Try matching by unique identifiers: name, id, or aria-label
			for attr_key in ['name', 'id', 'aria-label']:
				if attr_key in hist_attrs and hist_attrs[attr_key]:
					for idx, elem in selector_items:
						if (
							elem.node_name.lower() == hist_name
							and elem.attributes
							and elem.attributes.get(attr_key) == hist_attrs[attr_key]
						):
							highlight_index = idx
							match_level = MatchLevel.ATTRIBUTE
							self.logger.info(f'Element matched via {attr_key} attribute: {hist_attrs[attr_key]}')
							break
					if highlight_index is not None:
						break

			if highlight_index is None:
				tried_attrs = [k for k in ['name', 'id', 'aria-label'] if k in hist_attrs and hist_attrs[k]]
				# Log what was tried and what's available on the page for debugging
				same_node_elements = [
					(idx, elem.attributes.get('aria-label') or elem.attributes.get('id') or elem.attributes.get('name'))
					for idx, elem in selector_items
					if elem.node_name.lower() == hist_name and elem.attributes
				]
				self.logger.info(
					f'🔍 ATTRIBUTE match failed for <{hist_name.upper()}> '
					f'(tried: {tried_attrs}, looking for: {[hist_attrs.get(k) for k in tried_attrs]}). '
					f'Page has {len(same_node_elements)} <{hist_name.upper()}> elements with identifiers: '
					f'{same_node_elements[:5]}{"..." if len(same_node_elements) > 5 else ""}'
				)

		if highlight_index is None:
			return None

		old_index = action.get_index()
		if old_index != highlight_index:
			action.set_index(highlight_index)
			level_name = match_level.name if match_level else 'UNKNOWN'
			self.logger.info(f'Element index updated {old_index} → {highlight_index} (matched at {level_name} level)')

		return action

	def _format_element_for_error(self, elem: DOMInteractedElement | None) -> str:
		"""Format element info for error messages during history rerun."""
		if elem is None:
			return '<no element recorded>'

		parts = [f'<{elem.node_name}>']

		# Add key identifying attributes
		if elem.attributes:
			for key in ['name', 'id', 'aria-label', 'type']:
				if key in elem.attributes and elem.attributes[key]:
					parts.append(f'{key}="{elem.attributes[key]}"')

		# Add hash info
		parts.append(f'hash={elem.element_hash}')
		if elem.stable_hash:
			parts.append(f'stable_hash={elem.stable_hash}')

		# Add xpath (truncated)
		if elem.x_path:
			xpath_short = elem.x_path if len(elem.x_path) <= 60 else f'...{elem.x_path[-57:]}'
			parts.append(f'xpath="{xpath_short}"')

		return ' '.join(parts)

	def _is_redundant_retry_step(
		self,
		current_item: AgentHistory,
		previous_item: AgentHistory | None,
		previous_step_succeeded: bool,
	) -> bool:
		"""
		Detect if current step is a redundant retry of the previous step.

		This handles cases where the original run needed to click the same element multiple
		times due to slow page response, but during replay the first click already succeeded.
		When the page has already navigated, subsequent retry clicks on the same element
		would fail because that element no longer exists.

		Returns True if:
		- Previous step succeeded
		- Both steps target the same element (by element_hash, stable_hash, or xpath)
		- Both steps perform the same action type (e.g., both are clicks)
		"""
		if not previous_item or not previous_step_succeeded:
			return False

		# Get interacted elements from both steps (first action in each)
		curr_elements = current_item.state.interacted_element
		prev_elements = previous_item.state.interacted_element

		if not curr_elements or not prev_elements:
			return False

		curr_elem = curr_elements[0] if curr_elements else None
		prev_elem = prev_elements[0] if prev_elements else None

		if not curr_elem or not prev_elem:
			return False

		# Check if same element by various matching strategies
		same_by_hash = curr_elem.element_hash == prev_elem.element_hash
		same_by_stable_hash = (
			curr_elem.stable_hash is not None
			and prev_elem.stable_hash is not None
			and curr_elem.stable_hash == prev_elem.stable_hash
		)
		same_by_xpath = curr_elem.x_path == prev_elem.x_path

		if not (same_by_hash or same_by_stable_hash or same_by_xpath):
			return False

		# Check if same action type
		curr_actions = current_item.model_output.action if current_item.model_output else []
		prev_actions = previous_item.model_output.action if previous_item.model_output else []

		if not curr_actions or not prev_actions:
			return False

		# Get the action type (first key in the action dict)
		curr_action_data = curr_actions[0].model_dump(exclude_unset=True)
		prev_action_data = prev_actions[0].model_dump(exclude_unset=True)

		curr_action_type = next(iter(curr_action_data.keys()), None)
		prev_action_type = next(iter(prev_action_data.keys()), None)

		if curr_action_type != prev_action_type:
			return False

		self.logger.debug(
			f'🔄 Detected redundant retry: both steps target same element '
			f'<{curr_elem.node_name}> with action "{curr_action_type}"'
		)

		return True

	def _is_menu_opener_step(self, history_item: AgentHistory | None) -> bool:
		"""
		Detect if a step opens a dropdown/menu.

		Checks for common patterns indicating a menu opener:
		- Element has aria-haspopup attribute
		- Element has data-gw-click="toggleSubMenu" (Guidewire pattern)
		- Element has expand-button in class name
		- Element role is "menuitem" with aria-expanded

		Returns True if the step appears to open a dropdown/submenu.
		"""
		if not history_item or not history_item.state or not history_item.state.interacted_element:
			return False

		elem = history_item.state.interacted_element[0] if history_item.state.interacted_element else None
		if not elem:
			return False

		attrs = elem.attributes or {}

		# Check for common menu opener indicators
		if attrs.get('aria-haspopup') in ('true', 'menu', 'listbox'):
			return True
		if attrs.get('data-gw-click') == 'toggleSubMenu':
			return True
		if 'expand-button' in attrs.get('class', ''):
			return True
		if attrs.get('role') == 'menuitem' and attrs.get('aria-expanded') in ('false', 'true'):
			return True
		if attrs.get('role') == 'button' and attrs.get('aria-expanded') in ('false', 'true'):
			return True

		return False

	def _is_menu_item_element(self, elem: 'DOMInteractedElement | None') -> bool:
		"""
		Detect if an element is a menu item that appears inside a dropdown/menu.

		Checks for:
		- role="menuitem", "option", "menuitemcheckbox", "menuitemradio"
		- Element is inside a menu structure (has menu-related parent indicators)
		- ax_name is set (menu items typically have accessible names)

		Returns True if the element appears to be a menu item.
		"""
		if not elem:
			return False

		attrs = elem.attributes or {}

		# Check for menu item roles
		role = attrs.get('role', '')
		if role in ('menuitem', 'option', 'menuitemcheckbox', 'menuitemradio', 'treeitem'):
			return True

		# Elements in Guidewire menus have these patterns
		if 'gw-action--inner' in attrs.get('class', ''):
			return True
		if 'menuitem' in attrs.get('class', '').lower():
			return True

		# If element has an ax_name and looks like it could be in a menu
		# This is a softer check - only used if the previous step was a menu opener
		if elem.ax_name and elem.ax_name not in ('', None):
			# Common menu container classes
			elem_class = attrs.get('class', '').lower()
			if any(x in elem_class for x in ['dropdown', 'popup', 'menu', 'submenu', 'action']):
				return True

		return False

	async def _reexecute_menu_opener(
		self,
		opener_item: AgentHistory,
		ai_step_llm: 'BaseChatModel | None' = None,
	) -> bool:
		"""
		Re-execute a menu opener step to re-open a closed dropdown.

		This is used when a menu item can't be found because the dropdown
		closed during the wait between steps.

		Returns True if re-execution succeeded, False otherwise.
		"""
		try:
			self.logger.info('🔄 Re-opening dropdown/menu by re-executing previous step...')
			# Use a minimal delay - we want to quickly re-open the menu
			await self._execute_history_step(opener_item, delay=0.5, ai_step_llm=ai_step_llm, wait_for_elements=False)
			# Small delay to let the menu render
			await asyncio.sleep(0.3)
			return True
		except Exception as e:
			self.logger.warning(f'Failed to re-open dropdown: {e}')
			return False

	async def load_and_rerun(
		self,
		history_file: str | Path | None = None,
		variables: dict[str, str] | None = None,
		**kwargs,
	) -> list[ActionResult]:
		"""
		Load history from file and rerun it, optionally substituting variables.

		Args:
			history_file: Path to the history file
			variables: Optional dict mapping variable names to new values (e.g. {'email': 'new@example.com'})
			**kwargs: Additional arguments passed to rerun_history:
				- max_retries: Maximum retries per action (default: 3)
				- skip_failures: Continue on failure (default: True)
				- delay_between_actions: Delay when no saved interval (default: 2.0s)
				- max_step_interval: Cap on saved step_interval (default: 45.0s)
				- summary_llm: Custom LLM for final summary
				- ai_step_llm: Custom LLM for extract re-evaluation
		"""
		if not history_file:
			history_file = 'AgentHistory.json'
		history = AgentHistoryList.load_from_file(history_file, self.AgentOutput)

		# Substitute variables if provided
		if variables:
			history = self._substitute_variables_in_history(history, variables)

		return await self.rerun_history(history, **kwargs)

	def save_history(self, file_path: str | Path | None = None) -> None:
		"""Save the history to a file with sensitive data filtering"""
		if not file_path:
			file_path = 'AgentHistory.json'
		self.history.save_to_file(file_path, sensitive_data=self.sensitive_data)

	def export_chrome_recording(self, *, title: str = 'Browser Use recording') -> ChromeRecorderFlow:
		"""Export current action history as a Chrome DevTools Recorder flow."""

		return export_agent_history_to_chrome_recorder(self.history, title=title)

	def save_chrome_recording(self, file_path: str | Path | None = None, *, title: str = 'Browser Use recording') -> Path:
		"""Save current action history as a Chrome DevTools Recorder JSON file."""

		if not file_path:
			file_path = 'ChromeRecorder.json'
		return write_agent_history_chrome_recorder_flow(self.history, file_path, title=title)

	async def replay_chrome_recording(
		self,
		recording: str | Path | dict[str, Any] | ChromeRecorderFlow,
		*,
		max_steps: int | None = None,
		default_timeout_ms: int = 5000,
		stop_on_error: bool = True,
		skip_unsupported: bool = False,
		wait_after_step_ms: int = 0,
	) -> ChromeRecorderPlaybackResult:
		"""Replay a Chrome DevTools Recorder flow in the agent's current browser session."""

		await self.browser_session.start()
		player = ChromeRecorderPlayer(
			self.browser_session,
			default_timeout_ms=default_timeout_ms,
			stop_on_error=stop_on_error,
			skip_unsupported=skip_unsupported,
			wait_after_step_ms=wait_after_step_ms,
		)
		return await player.replay(recording, max_steps=max_steps)

	def pause(self) -> None:
		"""Pause the agent before the next step"""
		print('\n\n⏸️ Paused the agent and left the browser open.\n\tPress [Enter] to resume or [Ctrl+C] again to quit.')
		self.state.paused = True
		self._external_pause_event.clear()

	def resume(self) -> None:
		"""Resume the agent"""
		# TODO: Locally the browser got closed
		print('----------------------------------------------------------------------')
		print('▶️  Resuming agent execution where it left off...\n')
		self.state.paused = False
		self._external_pause_event.set()

	def stop(self) -> None:
		"""Stop the agent"""
		self.logger.info('⏹️ Agent stopping')
		self.state.stopped = True

		# Signal pause event to unblock any waiting code so it can check the stopped state
		self._external_pause_event.set()

		# Task stopped

	def _convert_initial_actions(self, actions: list[dict[str, dict[str, Any]]]) -> list[ActionModel]:
		"""Convert dictionary-based actions to ActionModel instances"""
		converted_actions = []
		action_model = self.ActionModel
		for action_dict in actions:
			# Each action_dict should have a single key-value pair
			action_name = next(iter(action_dict))
			params = action_dict[action_name]

			# Get the parameter model for this action from registry
			action_info = self.tools.registry.registry.actions[action_name]
			param_model = action_info.param_model

			# Create validated parameters using the appropriate param model
			validated_params = param_model(**params)

			# Create ActionModel instance with the validated parameters
			action_model = self.ActionModel(**{action_name: validated_params})
			converted_actions.append(action_model)

		return converted_actions

	def _verify_and_setup_llm(self):
		"""
		Verify that the LLM API keys are setup and the LLM API is responding properly.
		Also handles tool calling method detection if in auto mode.
		"""

		# Skip verification if already done
		if getattr(self.llm, '_verified_api_keys', None) is True or CONFIG.SKIP_LLM_API_KEY_VERIFICATION:
			setattr(self.llm, '_verified_api_keys', True)
			return True

	@property
	def message_manager(self) -> MessageManager:
		return self._message_manager

	async def close(self):
		"""Close all resources"""
		try:
			# Only close browser if keep_alive is False (or not set)
			if self.browser_session is not None:
				if not self.browser_session.browser_profile.keep_alive:
					# Kill the browser session - this dispatches BrowserStopEvent,
					# stops the EventBus with clear=True, and recreates a fresh EventBus
					await self.browser_session.kill()
				else:
					# keep_alive=True sessions shouldn't keep the event loop alive after agent.run()
					await self.browser_session.event_bus.stop(
						clear=False,
						timeout=_get_timeout('TIMEOUT_BrowserSessionEventBusStopOnAgentClose', 1.0),
					)
					try:
						self.browser_session.event_bus.event_queue = None
						self.browser_session.event_bus._on_idle = None
					except Exception:
						pass

			# Close skill service if configured
			if self.skill_service is not None:
				await self.skill_service.close()

			# Force garbage collection
			gc.collect()

			# Debug: Log remaining threads and asyncio tasks
			import threading

			threads = threading.enumerate()
			self.logger.debug(f'🧵 Remaining threads ({len(threads)}): {[t.name for t in threads]}')

			# Get all asyncio tasks
			tasks = asyncio.all_tasks(asyncio.get_event_loop())
			# Filter out the current task (this close() coroutine)
			other_tasks = [t for t in tasks if t != asyncio.current_task()]
			if other_tasks:
				self.logger.debug(f'⚡ Remaining asyncio tasks ({len(other_tasks)}):')
				for task in other_tasks[:10]:  # Limit to first 10 to avoid spam
					self.logger.debug(f'  - {task.get_name()}: {task}')

		except Exception as e:
			self.logger.error(f'Error during cleanup: {e}')

	async def _update_action_models_for_page(self, page_url: str) -> None:
		"""Update action models with page-specific actions"""
		# Create new action model with current page's filtered actions
		self.ActionModel = self.tools.registry.create_action_model(page_url=page_url)
		# Update output model with the new actions
		if self.settings.flash_mode:
			self.AgentOutput = AgentOutput.type_with_custom_actions_flash_mode(self.ActionModel)
		elif self.settings.use_thinking:
			self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)
		else:
			self.AgentOutput = AgentOutput.type_with_custom_actions_no_thinking(self.ActionModel)

		# Update QA completion action model too.
		self.DoneActionModel = self.tools.registry.create_action_model(
			include_actions=[self._completion_action_name], page_url=page_url
		)
		if self.settings.flash_mode:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions_flash_mode(self.DoneActionModel)
		elif self.settings.use_thinking:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions(self.DoneActionModel)
		else:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions_no_thinking(self.DoneActionModel)

	async def authenticate_cloud_sync(self, show_instructions: bool = True) -> bool:
		"""
		Authenticate with cloud service for future runs.

		This is useful when users want to authenticate after a task has completed
		so that future runs will sync to the cloud.

		Args:
			show_instructions: Whether to show authentication instructions to user

		Returns:
			bool: True if authentication was successful
		"""
		self.logger.warning('Cloud sync has been removed and is no longer available')
		return False

	def run_sync(
		self,
		max_steps: int = 500,
		on_step_start: AgentHookFunc | None = None,
		on_step_end: AgentHookFunc | None = None,
	) -> AgentHistoryList[AgentStructuredOutput]:
		"""Synchronous wrapper around the async run method for easier usage without asyncio."""
		import asyncio

		return asyncio.run(self.run(max_steps=max_steps, on_step_start=on_step_start, on_step_end=on_step_end))

	def detect_variables(self) -> dict[str, DetectedVariable]:
		"""Detect reusable variables in agent history"""
		from browser_use.agent.variable_detector import detect_variables_in_history

		return detect_variables_in_history(self.history)

	def _substitute_variables_in_history(self, history: AgentHistoryList, variables: dict[str, str]) -> AgentHistoryList:
		"""Substitute variables in history with new values for rerunning with different data"""
		from browser_use.agent.variable_detector import detect_variables_in_history

		# Detect variables in the history
		detected_vars = detect_variables_in_history(history)

		# Build a mapping of original values to new values
		value_replacements: dict[str, str] = {}
		for var_name, new_value in variables.items():
			if var_name in detected_vars:
				old_value = detected_vars[var_name].original_value
				value_replacements[old_value] = new_value
			else:
				self.logger.warning(f'Variable "{var_name}" not found in history, skipping substitution')

		if not value_replacements:
			self.logger.info('No variables to substitute')
			return history

		# Create a deep copy of history to avoid modifying the original
		import copy

		modified_history = copy.deepcopy(history)

		# Substitute values in all actions
		substitution_count = 0
		for history_item in modified_history.history:
			if not history_item.model_output or not history_item.model_output.action:
				continue

			for action in history_item.model_output.action:
				# Handle both Pydantic models and dicts
				if hasattr(action, 'model_dump'):
					action_dict = action.model_dump()
				elif isinstance(action, dict):
					action_dict = action
				else:
					action_dict = vars(action) if hasattr(action, '__dict__') else {}

				# Substitute in all string fields
				substitution_count += self._substitute_in_dict(action_dict, value_replacements)

				# Update the action with modified values
				if hasattr(action, 'model_dump'):
					# For Pydantic RootModel, we need to recreate from the modified dict
					if hasattr(action, 'root'):
						# This is a RootModel - recreate it from the modified dict
						new_action = type(action).model_validate(action_dict)
						# Replace the root field in-place using object.__setattr__ to bypass Pydantic's immutability
						object.__setattr__(action, 'root', getattr(new_action, 'root'))
					else:
						# Regular Pydantic model - update fields in-place
						for key, val in action_dict.items():
							if hasattr(action, key):
								setattr(action, key, val)
				elif isinstance(action, dict):
					action.update(action_dict)

		self.logger.info(f'Substituted {substitution_count} value(s) in {len(value_replacements)} variable type(s) in history')
		return modified_history

	def _substitute_in_dict(self, data: dict, replacements: dict[str, str]) -> int:
		"""Recursively substitute values in a dictionary, returns count of substitutions made"""
		count = 0
		for key, value in data.items():
			if isinstance(value, str):
				# Replace if exact match
				if value in replacements:
					data[key] = replacements[value]
					count += 1
			elif isinstance(value, dict):
				# Recurse into nested dicts
				count += self._substitute_in_dict(value, replacements)
			elif isinstance(value, list):
				# Handle lists
				for i, item in enumerate(value):
					if isinstance(item, str) and item in replacements:
						value[i] = replacements[item]
						count += 1
					elif isinstance(item, dict):
						count += self._substitute_in_dict(item, replacements)
		return count


_PythonAgent = Agent
