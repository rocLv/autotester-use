import asyncio
import json
from pathlib import Path

import pytest
from pytest_httpserver import HTTPServer
from werkzeug import Response

from browser_use.browser import ApiCollectionConfig, BrowserSession
from browser_use.browser.api_collection import ApiSchemaCollector, normalize_api_path, same_registrable_domain
from browser_use.browser.profile import BrowserProfile


def test_same_registrable_domain_helper():
	assert same_registrable_domain('https://a.baidu.com/page', 'https://api.baidu.com/users')
	assert not same_registrable_domain('https://baidu.com', 'https://google.com')
	assert same_registrable_domain('https://a.example.co.uk', 'https://api.example.co.uk')


def test_normalize_api_path_ids():
	assert normalize_api_path('/api/users/123/orders/550e8400-e29b-41d4-a716-446655440000')[0] == ('/api/users/{id}/orders/{id}')
	assert normalize_api_path('/api/assets/0123456789abcdef0123456789abcdef')[0] == '/api/assets/{param}'


def test_api_schema_collector_merges_json_observations():
	collector = ApiSchemaCollector()

	collector.record(
		url='https://api.baidu.com/api/users/123?active=true',
		method='POST',
		status_code=200,
		request_headers={'content-type': 'application/json', 'authorization': 'Bearer secret'},
		response_headers={'content-type': 'application/json'},
		request_body='{"name":"Ada","age":37}',
		response_body='{"ok":true,"id":123}',
		response_content_type='application/json',
	)
	collector.record(
		url='https://api.baidu.com/api/users/456?active=false',
		method='POST',
		status_code=200,
		request_headers={'content-type': 'application/json'},
		response_headers={'content-type': 'application/json'},
		request_body='{"name":"Grace"}',
		response_body='{"ok":true,"id":456,"email":"grace@example.com"}',
		response_content_type='application/json',
	)

	schema = collector.to_openapi()
	operation = schema['paths']['/api/users/{id}']['post']

	assert schema['openapi'] == '3.0.3'
	assert schema['servers'] == [{'url': 'https://api.baidu.com'}]
	assert {'name': 'id', 'in': 'path', 'required': True, 'schema': {'type': 'string'}} in operation['parameters']
	assert {'name': 'active', 'in': 'query', 'required': False, 'schema': {'type': 'boolean'}} in operation['parameters']
	assert operation['requestBody']['content']['application/json']['schema']['properties']['name']['type'] == 'string'
	assert operation['responses']['200']['content']['application/json']['schema']['properties']['email']['type'] == 'string'


def test_api_schema_collector_keeps_sensitive_headers_out_by_default():
	collector = ApiSchemaCollector(config=ApiCollectionConfig(include_request_headers=True, include_response_headers=True))
	collector.record(
		url='https://api.baidu.com/api/session',
		method='GET',
		status_code=200,
		request_headers={'authorization': 'Bearer secret', 'x-client': 'browser-use'},
		response_headers={'set-cookie': 'secret=1', 'x-trace-id': 'abc'},
		response_body='{"ok":true}',
		response_content_type='application/json',
	)

	operation = collector.to_openapi()['paths']['/api/session']['get']
	header_parameters = [param for param in operation['parameters'] if param['in'] == 'header']
	response_headers = operation['responses']['200']['headers']

	assert [param['name'] for param in header_parameters] == ['x-client']
	assert list(response_headers) == ['x-trace-id']


@pytest.fixture(scope='function')
def api_http_server():
	server = HTTPServer()
	server.start()

	server.expect_request('/').respond_with_data(
		"""
		<!DOCTYPE html>
		<html>
			<head><title>API Collection</title></head>
			<body>
				<script>
					fetch('/api/users/123?active=true', {
						method: 'POST',
						headers: {'content-type': 'application/json'},
						body: JSON.stringify({name: 'Ada'})
					});
					fetch('/static/app.js');
				</script>
				<h1>API Collection</h1>
			</body>
		</html>
		""",
		content_type='text/html',
	)
	server.expect_request('/api/users/123').respond_with_handler(
		lambda request: Response(
			json.dumps({'id': 123, 'name': 'Ada'}),
			content_type='application/json',
		)
	)
	server.expect_request('/static/app.js').respond_with_data('console.log("ignored");', content_type='application/javascript')

	yield server
	server.stop()


@pytest.mark.asyncio
async def test_browser_session_collects_api_schema_to_file(api_http_server, tmp_path):
	output_path = tmp_path / 'openapi.json'
	session = BrowserSession(
		browser_profile=BrowserProfile(
			headless=True,
			user_data_dir=None,
			keep_alive=True,
			enable_default_extensions=False,
		)
	)
	await session.start()

	try:
		await session.start_api_collection(ApiCollectionConfig(output_path=output_path))
		await session.navigate_to(f'http://{api_http_server.host}:{api_http_server.port}/')
		await asyncio.sleep(0.5)

		exported_path = await session.export_api_schema()
		assert exported_path is not None
		schema = json.loads(Path(exported_path).read_text(encoding='utf-8'))

		assert exported_path == output_path
		assert schema['openapi'] == '3.0.3'
		assert '/api/users/{id}' in schema['paths']
		assert '/static/app.js' not in schema['paths']
		assert schema['paths']['/api/users/{id}']['post']['responses']['200']['content']['application/json']
	finally:
		await session.kill()
