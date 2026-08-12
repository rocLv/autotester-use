"""Fill the JJEBank registration form and complete its slider verification.

This demo operates the existing registration page. It intentionally does not
request an SMS code and does not click "立即注册", so running it does not create
an account. Use it only against an environment you are authorized to test.

Usage:

    BROWSER_USE_API_KEY=... uv run python examples/qa/jjebank_register_slider_qa.py
    BROWSER_USE_API_KEY=... uv run python examples/qa/jjebank_register_slider_qa.py --headed

Optional test data:

    REGISTER_PHONE=13800138000
    REGISTER_PASSWORD='QaDemo@123456'
"""

import argparse
import asyncio
import base64
import io
import math
import os
import sys
from collections.abc import Sequence

from dotenv import load_dotenv
from PIL import Image, ImageFilter
from pydantic import BaseModel, Field

from browser_use import (
	ActionResult,
	Agent,
	BrowserProfile,
	BrowserSession,
	ChatBrowserUse,
	QARunStatus,
	Tools,
)
from browser_use.actor.element import Element

REGISTER_URL = 'https://scbp.jjebank.cn:8888/#/register'
REGISTER_DOMAIN = 'scbp.jjebank.cn'
FORM_EXPECTATION = '三个输入框均已填写，确认密码与密码一致，页面没有显示这三个字段的格式或必填错误。'
SLIDER_EXPECTATION = '滑块区域明确显示“验证成功”。'


class SolveRegistrationSliderParams(BaseModel):
	"""Typed input for the site-specific slider action."""

	expected_text: str = Field(default='验证成功', min_length=1)


class FillRegistrationFormParams(BaseModel):
	"""Typed sensitive inputs for the site-specific registration form action."""

	# Length and format are verified after sensitive placeholders are resolved.
	# Schema-level maxima would reject the longer <secret>NAME</secret> tokens
	# before the Tools registry gets a chance to replace them.
	phone: str = Field(min_length=1)
	password: str = Field(min_length=1)


class SliderImageState(BaseModel):
	"""Observable DOM and image state needed to solve one slider challenge."""

	background_src: str
	piece_src: str
	canvas_css_width: float = Field(gt=60)
	canvas_css_height: float = Field(gt=0)


class FormVerification(BaseModel):
	"""Non-sensitive form state observed after browser input."""

	phone_valid: bool
	password_valid: bool
	confirmation_matches: bool
	all_fields_nonempty: bool

	@property
	def passed(self) -> bool:
		"""Whether every local form constraint is satisfied."""

		return all(self.model_dump().values())


class SliderEstimate(BaseModel):
	"""Auditable output from image matching and coordinate conversion."""

	gap_x_image_px: int = Field(ge=0)
	target_left_css_px: float = Field(ge=0)
	drag_distance_css_px: float = Field(ge=0)
	match_score: float


def build_task() -> str:
	"""Build an explicit, non-submitting QA task for the real registration page."""

	return f"""打开 {REGISTER_URL}，测试企业用户注册表单和滑动验证。

测试数据：手机号使用敏感数据 REGISTER_PHONE；密码和确认密码都使用敏感数据 REGISTER_PASSWORD。
前置条件：无。起始 URL 导航和敏感数据注入由 QA Runner 管理，不作为业务前置条件。

严格执行以下业务步骤：
1. 调用 fill_registration_form 动作填写注册表单：phone 使用敏感数据 REGISTER_PHONE，password 使用敏感数据 REGISTER_PASSWORD。不要点击“获取验证码”或“立即注册”。
   预期结果：{FORM_EXPECTATION}
2. 调用 solve_registration_slider 动作完成拼图滑动验证。不要用 JavaScript 修改 Vue 状态、DOM class 或隐藏字段来伪造成功。
   预期结果：{SLIDER_EXPECTATION}

安全约束：完成第二步后立即结束测试；不得请求短信验证码，不得填写短信验证码，不得点击“立即注册”。
"""


def _decode_data_url(source: str) -> Image.Image:
	"""Decode a browser image data URL into an RGBA Pillow image."""

	header, separator, payload = source.partition(',')
	if not separator or ';base64' not in header or not header.startswith('data:image/'):
		raise ValueError('Slider image source is not a base64 image data URL')
	try:
		return Image.open(io.BytesIO(base64.b64decode(payload, validate=True))).convert('RGBA')
	except Exception as exc:
		raise ValueError(f'Unable to decode slider image: {exc}') from exc


def _alpha_mask(piece: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
	"""Extract the visible puzzle mask and its bounding box."""

	alpha = piece.getchannel('A').point(lambda value: 255 if value >= 32 else 0)
	bbox = alpha.getbbox()
	if bbox is None:
		raise ValueError('Puzzle-piece image has no visible pixels')

	# Some services return an opaque RGB canvas. In that case, infer foreground
	# from the difference to the four corner colors before giving up.
	if bbox == (0, 0, piece.width, piece.height) and alpha.getextrema() == (255, 255):
		rgb = piece.convert('RGB')
		corners = [rgb.getpixel((0, 0)), rgb.getpixel((piece.width - 1, 0)), rgb.getpixel((0, piece.height - 1))]
		background = tuple(sum(color[channel] for color in corners) // len(corners) for channel in range(3))
		inferred = Image.new('L', piece.size, 0)
		inferred_pixels = inferred.load()
		for y in range(piece.height):
			for x in range(piece.width):
				pixel = rgb.getpixel((x, y))
				if sum(abs(pixel[channel] - background[channel]) for channel in range(3)) > 36:
					inferred_pixels[x, y] = 255
		inferred_bbox = inferred.getbbox()
		if inferred_bbox is not None:
			alpha = inferred
			bbox = inferred_bbox

	return alpha.crop(bbox), bbox


def _boundary_points(mask: Image.Image) -> list[tuple[int, int]]:
	"""Return foreground pixels on the puzzle contour."""

	pixels = mask.load()
	points: list[tuple[int, int]] = []
	for y in range(mask.height):
		for x in range(mask.width):
			if pixels[x, y] == 0:
				continue
			if (
				x == 0
				or y == 0
				or x == mask.width - 1
				or y == mask.height - 1
				or pixels[x - 1, y] == 0
				or pixels[x + 1, y] == 0
				or pixels[x, y - 1] == 0
				or pixels[x, y + 1] == 0
			):
				points.append((x, y))
	if not points:
		raise ValueError('Puzzle-piece contour could not be extracted')
	return points


def estimate_gap_offset(background: Image.Image, piece: Image.Image) -> tuple[int, float]:
	"""Estimate the missing-piece x coordinate from image edge and color evidence."""

	background_rgba = background.convert('RGBA')
	piece_rgba = piece.convert('RGBA')
	mask, bbox = _alpha_mask(piece_rgba)
	left, top, _, _ = bbox
	piece_crop = piece_rgba.crop(bbox).convert('L')
	background_gray = background_rgba.convert('L')
	background_edges = background_gray.filter(ImageFilter.FIND_EDGES)
	boundary = _boundary_points(mask)
	foreground = [(x, y) for y in range(mask.height) for x in range(mask.width) if mask.getpixel((x, y)) != 0]

	if top + mask.height > background_gray.height:
		raise ValueError('Puzzle-piece image is taller than the background challenge')

	minimum_x = max(35, -left)
	maximum_x = background_gray.width - mask.width - 2
	if maximum_x <= minimum_x:
		raise ValueError('Slider challenge dimensions do not leave a searchable horizontal range')

	edge_pixels = background_edges.load()
	background_pixels = background_gray.load()
	piece_pixels = piece_crop.load()
	best_x = minimum_x
	best_score = -math.inf
	for candidate_x in range(minimum_x, maximum_x + 1):
		edge_score = sum(edge_pixels[candidate_x + x, top + y] for x, y in boundary) / len(boundary)
		color_error = sum(abs(background_pixels[candidate_x + x, top + y] - piece_pixels[x, y]) for x, y in foreground) / len(
			foreground
		)
		# This site's challenge replaces the missing region with a dark cut-out.
		# The correct position therefore has both a matching contour and a large
		# color difference from the original puzzle-piece pixels.
		score = edge_score * 1.2 + color_error * 0.8
		if score > best_score:
			best_x = candidate_x
			best_score = score

	return best_x - left, best_score


async def _first_visible(elements: Sequence[Element]) -> tuple[Element, dict[str, float]]:
	"""Return the first element with a non-empty viewport bounding box."""

	for element in elements:
		box = await element.get_bounding_box()
		if box and box['width'] > 0 and box['height'] > 0:
			return element, box
	raise RuntimeError('No visible slider button was found')


async def _replace_focused_text(browser_session: BrowserSession, element: Element, value: str) -> None:
	"""Replace one field through real CDP input without per-character DOM staleness."""

	page = await browser_session.must_get_current_page()
	session_id = await page.session_id
	await element.click()
	select_all_modifier = 4 if sys.platform == 'darwin' else 2
	for event_type in ('keyDown', 'keyUp'):
		await browser_session.cdp_client.send.Input.dispatchKeyEvent(
			{
				'type': event_type,
				'key': 'a',
				'code': 'KeyA',
				'windowsVirtualKeyCode': 65,
				'modifiers': select_all_modifier,
			},
			session_id=session_id,
		)
	for event_type in ('keyDown', 'keyUp'):
		await browser_session.cdp_client.send.Input.dispatchKeyEvent(
			{
				'type': event_type,
				'key': 'Backspace',
				'code': 'Backspace',
				'windowsVirtualKeyCode': 8,
			},
			session_id=session_id,
		)
	await browser_session.cdp_client.send.Input.insertText({'text': value}, session_id=session_id)
	await asyncio.sleep(0.1)


async def _wait_for_slider_images(browser_session: BrowserSession) -> SliderImageState:
	"""Wait for the challenge image pair after the slider popover opens."""

	page = await browser_session.must_get_current_page()
	for _ in range(30):
		raw_state = await page.evaluate(
			"""() => {
				const visible = (selector) => Array.from(document.querySelectorAll(selector)).find((element) => {
					const rect = element.getBoundingClientRect();
					const style = window.getComputedStyle(element);
					return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
				});
				const background = visible('.slide-canvas');
				const piece = visible('.slide-block');
				if (!background || !piece || !background.src || !piece.src || !background.complete || !piece.complete) {
					return null;
				}
				const rect = background.getBoundingClientRect();
				return {
					background_src: background.src,
					piece_src: piece.src,
					canvas_css_width: rect.width,
					canvas_css_height: rect.height,
				};
			}"""
		)
		if raw_state:
			try:
				return SliderImageState.model_validate_json(raw_state)
			except Exception:
				pass
		await asyncio.sleep(0.1)
	else:
		raise RuntimeError('Slider puzzle images did not become observable')


async def _drag_slider(browser_session: BrowserSession, distance: float) -> None:
	"""Perform a genuine pointer drag with a smooth, reproducible trajectory."""

	page = await browser_session.must_get_current_page()
	buttons = await page.get_elements_by_css_selector('.slider-button')
	_, box = await _first_visible(buttons)
	mouse = await page.mouse
	start_x = round(box['x'] + box['width'] / 2)
	start_y = round(box['y'] + box['height'] / 2)
	await mouse.move(start_x, start_y)
	await browser_session.cdp_client.send.Input.dispatchMouseEvent(
		{'type': 'mousePressed', 'x': start_x, 'y': start_y, 'button': 'left', 'clickCount': 1},
		session_id=await page.session_id,
	)

	steps = max(18, min(36, round(distance / 7)))
	for step in range(1, steps + 1):
		progress = step / steps
		eased_progress = 1 - (1 - progress) ** 2
		x = start_x + distance * eased_progress
		y = start_y + math.sin(progress * math.pi * 2) * 1.5
		await browser_session.cdp_client.send.Input.dispatchMouseEvent(
			{'type': 'mouseMoved', 'x': x, 'y': y, 'button': 'left'},
			session_id=await page.session_id,
		)
		await asyncio.sleep(0.018)

	await browser_session.cdp_client.send.Input.dispatchMouseEvent(
		{'type': 'mouseReleased', 'x': start_x + distance, 'y': start_y, 'button': 'left', 'clickCount': 1},
		session_id=await page.session_id,
	)


async def _slider_succeeded(browser_session: BrowserSession, expected_text: str) -> bool:
	"""Poll the site's own visible success state after server verification."""

	page = await browser_session.must_get_current_page()
	for _ in range(30):
		result = await page.evaluate(
			"""(expectedText) => {
				const success = Array.from(document.querySelectorAll('.success-tips')).find((element) => {
					const rect = element.getBoundingClientRect();
					return rect.width > 0 && rect.height > 0 && element.textContent.includes(expectedText);
				});
				return Boolean(success && success.closest('.verify-success'));
			}""",
			expected_text,
		)
		if result.lower() == 'true':
			return True
		await asyncio.sleep(0.1)
	return False


tools = Tools()


@tools.action(
	'Fill the visible JJEBank registration phone, password, and confirmation fields. This site re-renders its password input after the first character, so use this action instead of generic per-character input. It verifies values without returning sensitive text and never requests SMS or submits registration.',
	param_model=FillRegistrationFormParams,
	allowed_domains=[REGISTER_DOMAIN],
)
async def fill_registration_form(
	params: FillRegistrationFormParams,
	browser_session: BrowserSession,
) -> ActionResult:
	"""Fill the three fields with genuine input events and verify site constraints."""

	try:
		page = await browser_session.must_get_current_page()
		phone_inputs = await page.get_elements_by_css_selector('input[placeholder="请输入手机号码"]')
		password_inputs = await page.get_elements_by_css_selector('input[placeholder="请输入密码"]')
		phone_element, _ = await _first_visible(phone_inputs)
		visible_passwords: list[Element] = []
		for password_input in password_inputs:
			box = await password_input.get_bounding_box()
			if box and box['width'] > 0 and box['height'] > 0:
				visible_passwords.append(password_input)
		if len(visible_passwords) != 2:
			raise RuntimeError(f'Expected two visible password fields, found {len(visible_passwords)}')

		await _replace_focused_text(browser_session, phone_element, params.phone)
		# Re-query before each password because this page replaces the underlying
		# input node when its password visibility/type state changes.
		for password_index in range(2):
			password_inputs = await page.get_elements_by_css_selector('input[placeholder="请输入密码"]')
			visible_passwords = []
			for password_input in password_inputs:
				box = await password_input.get_bounding_box()
				if box and box['width'] > 0 and box['height'] > 0:
					visible_passwords.append(password_input)
			if len(visible_passwords) != 2:
				raise RuntimeError(f'Expected two visible password fields, found {len(visible_passwords)}')
			await _replace_focused_text(browser_session, visible_passwords[password_index], params.password)

		await page.evaluate(
			"""() => {
				if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
				return true;
			}"""
		)
		await asyncio.sleep(0.2)
		verification_raw = await page.evaluate(
			"""() => {
				const visible = (selector) => Array.from(document.querySelectorAll(selector)).filter((element) => {
					const rect = element.getBoundingClientRect();
					return rect.width > 0 && rect.height > 0;
				});
				const phone = visible('input[placeholder="请输入手机号码"]')[0];
				const passwords = visible('input[placeholder="请输入密码"]');
				const passwordRule = sessionStorage.getItem('passwordRule');
				const passwordPattern = passwordRule ? new RegExp(passwordRule) : /^(?=.*[A-Za-z])(?=.*\\d)(?=.*[!?@#$%^&*()_+])[A-Za-z\\d!?@#$%^&*()_+]{8,16}$/;
				return {
					phone_valid: Boolean(phone && /^(13|14|15|16|17|18|19)\\d{9}$/.test(phone.value)),
					password_valid: Boolean(passwords[0] && passwordPattern.test(passwords[0].value)),
					confirmation_matches: Boolean(passwords.length === 2 && passwords[0].value === passwords[1].value),
					all_fields_nonempty: Boolean(phone && phone.value && passwords.length === 2 && passwords.every((item) => item.value)),
				};
			}"""
		)
		verification = FormVerification.model_validate_json(verification_raw)
		if not verification.passed:
			return ActionResult(
				error='The site-specific input action completed, but one or more registration field constraints were not met.',
				metadata={
					'form_verification': verification.model_dump(),
					'qa_target_proof': {
						'target_name': 'registration phone, password, and confirmation fields',
						'target_matched': False,
						'verification': verification.model_dump(),
					},
					'qa_expectation_proof': {
						'requirement_quote': FORM_EXPECTATION,
						'expectation_met': False,
						'verification': verification.model_dump(),
					},
				},
			)
		return ActionResult(
			extracted_content='注册手机号、密码和确认密码已填写；页面真实值通过格式、一致性和非空检查。',
			long_term_memory='Registration phone, password, and confirmation were filled and locally validated without exposing values.',
			metadata={
				'form_verification': verification.model_dump(),
				'qa_target_proof': {
					'target_name': 'registration phone, password, and confirmation fields',
					'target_matched': True,
					'verification': verification.model_dump(),
				},
				'qa_expectation_proof': {
					'requirement_quote': FORM_EXPECTATION,
					'expectation_met': True,
					'verification': verification.model_dump(),
				},
			},
		)
	except Exception as exc:
		return ActionResult(error=f'Unable to fill the registration form: {type(exc).__name__}: {exc}')


@tools.action(
	'Complete the visible puzzle slider on the JJEBank registration page with a genuine mouse drag, then verify the site-owned “验证成功” state. Call this only after phone, password, and confirmation password are valid.',
	param_model=SolveRegistrationSliderParams,
	allowed_domains=[REGISTER_DOMAIN],
)
async def solve_registration_slider(
	params: SolveRegistrationSliderParams,
	browser_session: BrowserSession,
) -> ActionResult:
	"""Solve one authorized slider challenge without changing application state directly."""

	try:
		page = await browser_session.must_get_current_page()
		buttons = await page.get_elements_by_css_selector('.slider-button')
		_, button_box = await _first_visible(buttons)
		mouse = await page.mouse
		button_x = round(button_box['x'] + button_box['width'] / 2)
		button_y = round(button_box['y'] + button_box['height'] / 2)
		# The component opens on mouseover. Moving away first guarantees a fresh
		# mouseover after an earlier validation failure or retry.
		await mouse.move(max(1, button_x - 90), max(1, button_y - 55))
		await asyncio.sleep(0.08)
		await mouse.move(
			button_x,
			button_y,
		)
		state = await _wait_for_slider_images(browser_session)
		background = _decode_data_url(state.background_src)
		piece = _decode_data_url(state.piece_src)
		gap_x, score = estimate_gap_offset(background, piece)
		target_left_css = gap_x * state.canvas_css_width / background.width
		drag_distance = target_left_css * (state.canvas_css_width - 40) / (state.canvas_css_width - 60)
		drag_distance = max(1.0, min(state.canvas_css_width - 41, drag_distance))
		estimate = SliderEstimate(
			gap_x_image_px=gap_x,
			target_left_css_px=target_left_css,
			drag_distance_css_px=drag_distance,
			match_score=score,
		)

		await _drag_slider(browser_session, drag_distance)
		if not await _slider_succeeded(browser_session, params.expected_text):
			return ActionResult(
				error='The genuine drag completed, but the site did not show the expected slider success state.',
				metadata={
					'slider_estimate': estimate.model_dump(),
					'site_success_observed': False,
					'qa_target_proof': {
						'target_name': 'registration puzzle slider',
						'target_matched': False,
						'verification': {'site_success_observed': False},
					},
					'qa_expectation_proof': {
						'requirement_quote': SLIDER_EXPECTATION,
						'expectation_met': False,
						'verification': {'site_success_observed': False},
					},
				},
			)

		return ActionResult(
			extracted_content=f'滑动验证已完成，页面可见“{params.expected_text}”。',
			long_term_memory=f'JJEBank registration slider visibly showed “{params.expected_text}”.',
			metadata={
				'slider_estimate': estimate.model_dump(),
				'site_success_observed': True,
				'qa_target_proof': {
					'target_name': 'registration puzzle slider',
					'target_matched': True,
					'verification': {'site_success_observed': True},
				},
				'qa_expectation_proof': {
					'requirement_quote': SLIDER_EXPECTATION,
					'expectation_met': True,
					'verification': {'site_success_observed': True},
				},
			},
		)
	except Exception as exc:
		return ActionResult(error=f'Unable to complete the registration slider: {type(exc).__name__}: {exc}')


def build_browser(*, headed: bool) -> BrowserSession:
	"""Create a browser restricted to the registration host."""

	return BrowserSession(
		browser_profile=BrowserProfile(
			headless=not headed,
			user_data_dir=None,
			keep_alive=False,
			allowed_domains=[REGISTER_DOMAIN],
		)
	)


async def run_demo(*, headed: bool, full_result: bool = False) -> int:
	"""Run the form-fill and slider QA case and print its typed result."""

	phone = os.getenv('REGISTER_PHONE', '13800138000')
	password = os.getenv('REGISTER_PASSWORD', 'QaDemo@123456')
	browser = build_browser(headed=headed)
	agent = Agent(
		task=build_task(),
		llm=ChatBrowserUse(),
		browser=browser,
		tools=tools,
		sensitive_data={
			REGISTER_DOMAIN: {
				'REGISTER_PHONE': phone,
				'REGISTER_PASSWORD': password,
			}
		},
		max_agent_retries_per_step=2,
		reuse_compiled_test_case=True,
		reuse_login_state=True,
		use_vision=False,
	)
	try:
		history = await agent.run(max_steps=40)
		result = history.qa_result
		if result is None:
			print('Agent returned no qa_result.')
			return 2
		if result.test_case is not None:
			print('\n强结构化测试用例：')
			print(result.test_case.to_markdown_table())
		print('\nQA 结果：')
		print(f'状态: {result.status.value}')
		print(f'摘要: {result.summary}')
		print(f'LLM 调用数: {result.llm_call_count}')
		for step_result in result.step_results:
			actual_result = step_result.judgement.actual_result if step_result.judgement else '无裁决结果'
			print(f'- {step_result.step.step_id}: {step_result.status.value} — {actual_result}')
		if full_result:
			print('\n完整 QARunResult JSON：')
			print(result.model_dump_json(indent=2))
		return 0 if result.status == QARunStatus.PASSED else 1
	finally:
		await browser.kill()


def parse_args() -> argparse.Namespace:
	"""Parse demo options."""

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--headed', action='store_true', help='Show Chromium while the QA case executes')
	parser.add_argument('--full-result', action='store_true', help='Print the complete QARunResult JSON including evidence')
	return parser.parse_args()


def main() -> int:
	"""Run the authorized non-submitting registration demo."""

	load_dotenv()
	args = parse_args()
	return asyncio.run(run_demo(headed=args.headed, full_result=args.full_result))


if __name__ == '__main__':
	raise SystemExit(main())
