"""API collection helpers for building OpenAPI documents from browser traffic."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

from pydantic import BaseModel, ConfigDict, Field

SENSITIVE_HEADER_NAMES = {'authorization', 'cookie', 'set-cookie', 'proxy-authorization'}
COMMON_SECOND_LEVEL_PUBLIC_SUFFIXES = {
	'ac.uk',
	'co.jp',
	'co.kr',
	'co.nz',
	'co.uk',
	'com.au',
	'com.br',
	'com.cn',
	'com.hk',
	'com.mx',
	'com.sg',
	'com.tw',
	'com.tr',
	'net.cn',
	'org.cn',
	'gov.cn',
	'edu.cn',
	'net.au',
	'org.au',
}


class ApiCollectionConfig(BaseModel):
	"""Configuration for collecting same-site API traffic into an OpenAPI document."""

	output_path: str | Path | None = None
	include_request_headers: bool = False
	include_response_headers: bool = False
	include_examples: bool = False
	same_site_only: bool = True


class CollectedApiEndpoint(BaseModel):
	"""Merged information for one normalized API endpoint."""

	model_config = ConfigDict(arbitrary_types_allowed=True)

	method: str
	url: str
	path: str
	origin: str
	query_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
	path_params: set[str] = Field(default_factory=set)
	request_content_type: str | None = None
	response_content_type: str | None = None
	status_codes: dict[str, dict[str, Any]] = Field(default_factory=dict)
	request_schema: dict[str, Any] | None = None
	response_schemas: dict[str, dict[str, Any]] = Field(default_factory=dict)
	request_headers: dict[str, str] = Field(default_factory=dict)
	response_headers: dict[str, str] = Field(default_factory=dict)
	request_example: Any | None = None
	response_examples: dict[str, Any] = Field(default_factory=dict)


class OpenApiDocument(BaseModel):
	"""OpenAPI 3.0 document wrapper."""

	model_config = ConfigDict(extra='allow')

	openapi: str = '3.0.3'
	info: dict[str, Any]
	servers: list[dict[str, str]] = Field(default_factory=list)
	paths: dict[str, Any] = Field(default_factory=dict)


def registrable_domain(hostname: str | None) -> str | None:
	"""Return a lightweight registrable domain without adding a public suffix dependency."""
	if not hostname:
		return None
	host = hostname.strip('.').lower()
	if not host or host == 'localhost':
		return host or None
	if re.fullmatch(r'\d{1,3}(?:\.\d{1,3}){3}', host):
		return host

	parts = [part for part in host.split('.') if part]
	if len(parts) <= 2:
		return host

	suffix_2 = '.'.join(parts[-2:])
	if suffix_2 in COMMON_SECOND_LEVEL_PUBLIC_SUFFIXES and len(parts) >= 3:
		return '.'.join(parts[-3:])
	return '.'.join(parts[-2:])


def same_registrable_domain(first_url: str | None, second_url: str | None) -> bool:
	"""Check whether two URLs share a registrable domain."""
	first = registrable_domain(urlparse(first_url or '').hostname)
	second = registrable_domain(urlparse(second_url or '').hostname)
	return bool(first and second and first == second)


def normalize_api_path(path: str) -> tuple[str, set[str]]:
	"""Normalize common path identifiers into OpenAPI path parameters."""
	normalized_parts: list[str] = []
	path_params: set[str] = set()

	for raw_part in path.split('/'):
		if raw_part == '':
			continue
		part = raw_part
		param_name: str | None = None
		if re.fullmatch(r'\d+', part):
			param_name = 'id'
		elif re.fullmatch(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', part):
			param_name = 'id'
		elif re.fullmatch(r'[0-9a-fA-F]{16,}', part):
			param_name = 'param'

		if param_name:
			path_params.add(param_name)
			normalized_parts.append(f'{{{param_name}}}')
		else:
			normalized_parts.append(part)

	return '/' + '/'.join(normalized_parts), path_params


def is_json_content_type(content_type: str | None) -> bool:
	"""Return True when a MIME/content-type value represents JSON."""
	if not content_type:
		return False
	content_type = content_type.lower().split(';', 1)[0].strip()
	return content_type == 'application/json' or content_type.endswith('+json')


def parse_json_body(text: str | None) -> Any | None:
	"""Parse a JSON body, returning None when it is absent or invalid."""
	if not text:
		return None
	try:
		return json.loads(text)
	except Exception:
		return None


def infer_json_schema(value: Any) -> dict[str, Any]:
	"""Infer a compact JSON schema suitable for OpenAPI from a sample value."""
	if value is None:
		return {'nullable': True}
	if isinstance(value, bool):
		return {'type': 'boolean'}
	if isinstance(value, int) and not isinstance(value, bool):
		return {'type': 'integer'}
	if isinstance(value, float):
		return {'type': 'number'}
	if isinstance(value, str):
		return {'type': 'string'}
	if isinstance(value, list):
		if not value:
			return {'type': 'array', 'items': {}}
		item_schema = infer_json_schema(value[0])
		for item in value[1:]:
			item_schema = merge_json_schemas(item_schema, infer_json_schema(item))
		return {'type': 'array', 'items': item_schema}
	if isinstance(value, dict):
		properties: dict[str, Any] = {}
		required: list[str] = []
		for key, item in value.items():
			properties[str(key)] = infer_json_schema(item)
			if item is not None:
				required.append(str(key))
		schema: dict[str, Any] = {'type': 'object', 'properties': properties}
		if required:
			schema['required'] = sorted(required)
		return schema
	return {'type': 'string'}


def merge_json_schemas(first: dict[str, Any] | None, second: dict[str, Any] | None) -> dict[str, Any] | None:
	"""Merge two inferred schemas while preserving useful structure."""
	if first is None:
		return copy.deepcopy(second)
	if second is None:
		return copy.deepcopy(first)
	if first == second:
		return copy.deepcopy(first)

	first_type = first.get('type')
	second_type = second.get('type')
	if first_type != second_type:
		types = []
		for schema_type in (first_type, second_type):
			if isinstance(schema_type, list):
				types.extend(schema_type)
			elif schema_type:
				types.append(schema_type)
		return {'type': sorted(set(types))} if types else {}

	if first_type == 'object':
		properties: dict[str, Any] = {}
		for key in set(first.get('properties', {})) | set(second.get('properties', {})):
			properties[key] = merge_json_schemas(first.get('properties', {}).get(key), second.get('properties', {}).get(key))
		required = sorted(set(first.get('required', [])) & set(second.get('required', [])))
		merged: dict[str, Any] = {'type': 'object', 'properties': properties}
		if required:
			merged['required'] = required
		return merged

	if first_type == 'array':
		return {
			'type': 'array',
			'items': merge_json_schemas(first.get('items'), second.get('items')) or {},
		}

	return copy.deepcopy(first)


def safe_headers(headers: dict[str, str] | None) -> dict[str, str]:
	"""Return headers with sensitive values removed."""
	if not headers:
		return {}
	return {name.lower(): str(value) for name, value in headers.items() if name.lower() not in SENSITIVE_HEADER_NAMES}


class ApiSchemaCollector(BaseModel):
	"""Collects API request observations and exports an OpenAPI 3.0 document."""

	model_config = ConfigDict(arbitrary_types_allowed=True)

	config: ApiCollectionConfig = Field(default_factory=ApiCollectionConfig)
	endpoints: dict[str, CollectedApiEndpoint] = Field(default_factory=dict)
	servers: set[str] = Field(default_factory=set)
	title_domain: str | None = None

	def record(
		self,
		*,
		url: str,
		method: str,
		status_code: int | None = None,
		request_headers: dict[str, str] | None = None,
		response_headers: dict[str, str] | None = None,
		request_body: str | None = None,
		response_body: str | bytes | None = None,
		response_content_type: str | None = None,
	) -> None:
		"""Record one API request/response observation."""
		parsed = urlparse(url)
		if not parsed.scheme or not parsed.netloc:
			return

		origin = f'{parsed.scheme}://{parsed.netloc}'
		normalized_path, path_params = normalize_api_path(parsed.path or '/')
		method = method.lower()
		endpoint_key = f'{method} {origin} {normalized_path}'
		self.servers.add(origin)
		if self.title_domain is None:
			self.title_domain = registrable_domain(parsed.hostname)

		endpoint = self.endpoints.get(endpoint_key)
		if endpoint is None:
			endpoint = CollectedApiEndpoint(
				method=method,
				url=url,
				path=normalized_path,
				origin=origin,
			)
			self.endpoints[endpoint_key] = endpoint

		endpoint.path_params.update(path_params)
		for name, value in parse_qsl(parsed.query, keep_blank_values=True):
			endpoint.query_params.setdefault(
				name,
				{'name': name, 'in': 'query', 'required': False, 'schema': _schema_for_query_value(value)},
			)
		if self.config.include_request_headers:
			endpoint.request_headers.update(safe_headers(request_headers))
		if self.config.include_response_headers:
			endpoint.response_headers.update(safe_headers(response_headers))

		request_content_type = (request_headers or {}).get('content-type') or (request_headers or {}).get('Content-Type')
		if request_content_type:
			endpoint.request_content_type = request_content_type
		if response_content_type:
			endpoint.response_content_type = response_content_type

		request_json = parse_json_body(request_body)
		if request_json is not None:
			endpoint.request_schema = merge_json_schemas(endpoint.request_schema, infer_json_schema(request_json))
			if self.config.include_examples and endpoint.request_example is None:
				endpoint.request_example = _truncate_example(request_json)

		response_text: str | None
		if isinstance(response_body, bytes):
			response_text = response_body.decode('utf-8', errors='replace')
		else:
			response_text = response_body
		response_json = parse_json_body(response_text)
		status = str(status_code or 'default')
		if response_json is not None:
			endpoint.response_schemas[status] = merge_json_schemas(
				endpoint.response_schemas.get(status), infer_json_schema(response_json)
			) or {'type': 'object'}
			if self.config.include_examples and status not in endpoint.response_examples:
				endpoint.response_examples[status] = _truncate_example(response_json)

		endpoint.status_codes.setdefault(status, {'description': _status_description(status)})

	def to_openapi(self) -> dict[str, Any]:
		"""Build an OpenAPI 3.0 document from collected endpoints."""
		title = f'{self.title_domain or "Collected"} API'
		paths: dict[str, Any] = {}

		for endpoint in sorted(self.endpoints.values(), key=lambda item: (item.path, item.method)):
			operation: dict[str, Any] = {
				'summary': f'{endpoint.method.upper()} {endpoint.path}',
				'parameters': self._parameters_for(endpoint),
				'responses': self._responses_for(endpoint),
			}
			if endpoint.request_schema is not None:
				operation['requestBody'] = {
					'required': True,
					'content': {
						_content_type(endpoint.request_content_type): {
							'schema': endpoint.request_schema,
						}
					},
				}
				if self.config.include_examples and endpoint.request_example is not None:
					operation['requestBody']['content'][_content_type(endpoint.request_content_type)]['example'] = (
						endpoint.request_example
					)

			paths.setdefault(endpoint.path, {})[endpoint.method] = operation

		document = OpenApiDocument(
			info={'title': title, 'version': '1.0.0'},
			servers=[{'url': origin} for origin in sorted(self.servers)],
			paths=paths,
		)
		return document.model_dump(exclude_none=True, mode='json')

	def export(self, path: str | Path | None = None) -> Path | None:
		"""Write the OpenAPI document to disk."""
		resolved_path = path if path is not None else self.config.output_path
		if resolved_path is None:
			return None
		output_path = Path(resolved_path).expanduser().resolve()
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(json.dumps(self.to_openapi(), indent=2, ensure_ascii=False), encoding='utf-8')
		return output_path

	def _parameters_for(self, endpoint: CollectedApiEndpoint) -> list[dict[str, Any]]:
		parameters: list[dict[str, Any]] = []
		for name in sorted(endpoint.path_params):
			parameters.append({'name': name, 'in': 'path', 'required': True, 'schema': {'type': 'string'}})
		parameters.extend(endpoint.query_params[name] for name in sorted(endpoint.query_params))
		if self.config.include_request_headers:
			for name in sorted(endpoint.request_headers):
				parameters.append({'name': name, 'in': 'header', 'required': False, 'schema': {'type': 'string'}})
		return parameters

	def _responses_for(self, endpoint: CollectedApiEndpoint) -> dict[str, Any]:
		responses: dict[str, Any] = {}
		for status, metadata in sorted(endpoint.status_codes.items()):
			response: dict[str, Any] = {'description': metadata.get('description') or _status_description(status)}
			schema = endpoint.response_schemas.get(status)
			if schema is not None:
				content_type = _content_type(endpoint.response_content_type)
				response['content'] = {content_type: {'schema': schema}}
				if self.config.include_examples and status in endpoint.response_examples:
					response['content'][content_type]['example'] = endpoint.response_examples[status]
			if self.config.include_response_headers and endpoint.response_headers:
				response['headers'] = {
					name: {'schema': {'type': 'string'}, 'description': 'Observed response header'}
					for name in sorted(endpoint.response_headers)
				}
			responses[status] = response
		if not responses:
			responses['default'] = {'description': 'Observed response'}
		return responses


def _content_type(content_type: str | None) -> str:
	return content_type.split(';', 1)[0].strip() if content_type else 'application/json'


def _schema_for_query_value(value: str) -> dict[str, Any]:
	if value.lower() in {'true', 'false'}:
		return {'type': 'boolean'}
	if re.fullmatch(r'-?\d+', value):
		return {'type': 'integer'}
	if re.fullmatch(r'-?\d+\.\d+', value):
		return {'type': 'number'}
	return {'type': 'string'}


def _status_description(status: str) -> str:
	if status == 'default':
		return 'Observed response'
	return f'HTTP {status} response'


def _truncate_example(value: Any, max_chars: int = 4096) -> Any:
	text = json.dumps(value, ensure_ascii=False)
	if len(text) <= max_chars:
		return value
	return text[:max_chars] + '...'
