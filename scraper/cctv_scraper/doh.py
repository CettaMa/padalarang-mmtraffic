import socket
import threading
import time
from typing import Any

import requests
import urllib3.util.connection
from urllib3.util.connection import create_connection as _orig_create_connection


class DoHResolver:
    """Resolve hostnames via Cloudflare DNS-over-HTTPS when system DNS fails."""

    _cache: dict[str, tuple[str | None, float]] = {}
    _ttl = 300  # cache for 5 minutes
    _max_cache_size = 256
    _lock = threading.Lock()

    @classmethod
    def _set_cache(cls, hostname: str, ip: str | None) -> None:
        """Store lookup result with FIFO eviction if cache size exceeds limit."""
        if len(cls._cache) >= cls._max_cache_size and hostname not in cls._cache:
            first_key = next(iter(cls._cache))
            cls._cache.pop(first_key, None)
        cls._cache[hostname] = (ip, time.time())

    @classmethod
    def resolve(cls, hostname: str) -> str | None:
        with cls._lock:
            cached = cls._cache.get(hostname)
            if cached and time.time() - cached[1] < cls._ttl:
                return cached[0]

        try:
            socket.getaddrinfo(hostname, 443, socket.AF_INET)
            # System DNS works -> cache negative result (None) so we don't re-query
            with cls._lock:
                cls._set_cache(hostname, None)
            return None
        except socket.gaierror:
            pass

        try:
            resp = requests.get(
                "https://cloudflare-dns.com/dns-query",
                params={"name": hostname, "type": "A"},
                headers={"accept": "application/dns-json"},
                timeout=5,
            )
            for answer in resp.json().get("Answer", []):
                if answer.get("type") == 1:
                    ip = answer["data"]
                    with cls._lock:
                        cls._set_cache(hostname, ip)
                    return ip
        except (requests.RequestException, ValueError, KeyError):
            pass

        # If resolution fails, cache negative result for TTL to avoid repeated slow timeouts
        with cls._lock:
            cls._set_cache(hostname, None)
        return None


_doh_patched_hosts: dict[str, str] = {}
_doh_lock = threading.Lock()
_MAX_PATCHED_HOSTS = 256


def _doh_create_connection(address: tuple[str, int], *args: Any, **kwargs: Any) -> Any:
    host, port = address
    with _doh_lock:
        ip = _doh_patched_hosts.get(host)

    if not ip:
        ip = DoHResolver.resolve(host)
        if ip:
            with _doh_lock:
                if len(_doh_patched_hosts) >= _MAX_PATCHED_HOSTS and host not in _doh_patched_hosts:
                    first_key = next(iter(_doh_patched_hosts))
                    _doh_patched_hosts.pop(first_key, None)
                _doh_patched_hosts[host] = ip

    if ip:
        return _orig_create_connection((ip, port), *args, **kwargs)
    return _orig_create_connection(address, *args, **kwargs)


urllib3.util.connection.create_connection = _doh_create_connection
