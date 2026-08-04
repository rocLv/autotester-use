"""Navigation-scope helpers for Web UI QA agents."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

import tldextract
from pydantic import BaseModel, ConfigDict

_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)
_BROWSER_LIFECYCLE_URLS = {'about:blank', 'chrome://new-tab-page/', 'chrome://new-tab-page', 'chrome://newtab/'}


def _canonical_hostname(hostname: str) -> str:
	host = hostname.rstrip('.').lower()
	try:
		return host.encode('idna').decode('ascii')
	except UnicodeError as exc:
		raise ValueError(f'Invalid internationalized hostname: {hostname}') from exc


def registrable_domain(hostname: str) -> str:
	"""Return an offline-PSL registrable domain, preserving exact local/IP hosts."""

	host = _canonical_hostname(hostname)
	if host == 'localhost':
		return host
	try:
		ipaddress.ip_address(host.strip('[]'))
		return host
	except ValueError:
		pass

	extracted = _TLD_EXTRACTOR(host)
	domain = extracted.top_domain_under_public_suffix
	if not domain:
		raise ValueError(f'Cannot derive a registrable domain from hostname: {hostname}')
	return domain


class NavigationScope(BaseModel):
	"""Top-level navigation boundary derived from a task's explicit root URL."""

	model_config = ConfigDict(frozen=True)

	root_url: str
	root_host: str
	registrable_domain: str
	exact_host_only: bool = False

	@classmethod
	def from_root_url(cls, root_url: str) -> NavigationScope:
		parsed = urlparse(root_url)
		if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
			raise ValueError('QA root URL must be an absolute HTTP(S) URL')
		if parsed.username or parsed.password:
			raise ValueError('QA root URL must not contain credentials')
		host = _canonical_hostname(parsed.hostname)
		exact_host_only = host == 'localhost'
		try:
			ipaddress.ip_address(host.strip('[]'))
			exact_host_only = True
		except ValueError:
			pass
		return cls(
			root_url=root_url,
			root_host=host,
			registrable_domain=registrable_domain(host),
			exact_host_only=exact_host_only,
		)

	def allows(self, candidate_url: str) -> bool:
		"""Return whether a top-level candidate URL is inside this QA scope."""

		if candidate_url in _BROWSER_LIFECYCLE_URLS:
			return True
		parsed = urlparse(candidate_url)
		if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
			return False
		if parsed.username or parsed.password:
			return False
		try:
			host = _canonical_hostname(parsed.hostname)
		except ValueError:
			return False
		if self.exact_host_only:
			return host == self.root_host
		try:
			return registrable_domain(host) == self.registrable_domain
		except ValueError:
			return False
