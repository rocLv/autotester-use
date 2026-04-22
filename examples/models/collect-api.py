import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from browser_use import Agent, ApiCollectionConfig, Browser, ChatBrowserUse

URL = 'https://chat.zzbank.cn:9080/imclient/pages/index.html'


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Collect API calls while consulting ZZBank personal business.')
	parser.add_argument(
		'output',
		nargs='?',
		type=Path,
		default=Path(__file__).with_suffix('').with_name(f'{Path(__file__).stem}-apis.json'),
		help="Output OpenAPI JSON file. Defaults to '<python-file-name>-apis.json'.",
	)
	return parser.parse_args()


async def main():
	args = parse_args()
	output = args.output
	load_dotenv()

	browser = Browser(
		headless=False,
		keep_alive=True,
		disable_security=True,  # 兼容内网站点/自签名证书等情况
		user_data_dir=None,
	)

	await browser.start()

	try:
		await browser.start_api_collection(
			ApiCollectionConfig(
				output_path=output,
				same_site_only=True,
				include_request_headers=False,
				include_response_headers=False,
				include_examples=False,
			)
		)

		task = f"""
        1. 打开网页：{URL}
        2. 等待页面加载完成。
        3. 找到并进入“个人业务”。
        4. 在“个人业务”里发起咨询或开始对话。
        5. 如果需要输入咨询内容，就输入：“你好，我想咨询个人业务。”
        6. 如果页面出现无法点击的按钮，尝试使用键盘 Tab / Enter 完成操作。
        7. 完成进入个人业务咨询后，停止操作并说明当前页面状态。
        """

		agent = Agent(
			task=task,
			llm=ChatBrowserUse(),
			browser=browser,
			max_failures=5,
		)

		history = await agent.run(max_steps=30)

		await browser.export_api_schema(output)
		schema = browser.get_api_schema()

		print('\n任务完成')
		print(f'Agent final result: {history.final_result()}')
		print(f'OpenAPI 文件: {output.resolve()}')
		print(f'采集到 API paths 数量: {len(schema.get("paths", {}))}')

	finally:
		await browser.kill()


if __name__ == '__main__':
	asyncio.run(main())
