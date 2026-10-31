"""Dynamic model endpoint pools shared by pipeline components."""
from __future__ import annotations

import ipaddress
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError
from urllib.parse import urlparse, urlunparse
from urllib.request import ProxyHandler, Request, build_opener

import yaml


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EndpointPoolConfig:
    path: Path
    model: str | None
    base_urls: tuple[str, ...]
    wire_api: str = "chat_completions"
    api_key_env: str | None = None
    reasoning_effort: str | None = None
    store: bool = False
    trust_env: bool = True
    reload_on_request: bool = True
    failure_threshold: int = 1
    cooldown_seconds: float = 60.0
    load_aware: bool = False
    load_probe_interval_seconds: float = 5.0
    load_probe_timeout_seconds: float = 1.0
    max_waiting_requests: int = 0


@dataclass(frozen=True)
class EndpointLoad:
    running_requests: int
    waiting_requests: int
    pending_tokens: int
    utilization: float = 0.0


@dataclass
class _EndpointHealth:
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    last_error: str | None = None
    load: EndpointLoad | None = None
    load_checked_at: float = 0.0
    load_error: str | None = None


def normalize_base_urls(values: Iterable[str] | str | None) -> tuple[str, ...]:
    if not values:
        return ()
    items = values.split(",") if isinstance(values, str) else values
    normalized: list[str] = []
    for value in items:
        url = str(value).strip()
        if not url:
            continue
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
        url = url.rstrip("/")
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"invalid endpoint URL: {value!r}")
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError(f"invalid endpoint URL: {value!r}") from exc
        hostname = parsed.hostname
        if hostname.replace(".", "").isdigit():
            try:
                ipaddress.ip_address(hostname)
            except ValueError as exc:
                raise ValueError(f"invalid endpoint IP address: {hostname}") from exc
        normalized.append(url)
    return tuple(dict.fromkeys(normalized))


def resolve_endpoint_pool_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_yaml_mapping(path: str | Path) -> dict:
    return _load_yaml_mapping(Path(path).expanduser().resolve(), seen=())


def _load_yaml_mapping(path: Path, *, seen: tuple[Path, ...]) -> dict:
    if path in seen:
        chain = " -> ".join(str(item) for item in (*seen, path))
        raise ValueError(f"cyclic YAML extends chain: {chain}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    parents = raw.pop("extends", None)
    if not parents:
        return raw
    parent_values = [parents] if isinstance(parents, (str, Path)) else parents
    if not isinstance(parent_values, list):
        raise ValueError(f"config extends must be a path or list of paths: {path}")
    merged: dict = {}
    for value in parent_values:
        parent_path = resolve_endpoint_pool_path(value, path.parent)
        merged = _deep_merge(merged, _load_yaml_mapping(parent_path, seen=(*seen, path)))
    return _deep_merge(merged, raw)


def _deep_merge(parent: dict, child: dict) -> dict:
    merged = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_endpoint_pool_config(path: str | Path) -> EndpointPoolConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"endpoint pool must be a YAML mapping: {config_path}")
    if int(raw.get("schema_version", 1)) != 1:
        raise ValueError(f"unsupported endpoint pool schema_version: {config_path}")
    base_urls = normalize_base_urls(raw.get("base_urls"))
    if not base_urls:
        raise ValueError(f"endpoint pool base_urls must not be empty: {config_path}")
    routing = raw.get("routing") or {}
    if not isinstance(routing, dict):
        raise ValueError(f"endpoint pool routing must be a mapping: {config_path}")
    failure_threshold = int(routing.get("failure_threshold", 1))
    cooldown_seconds = float(routing.get("cooldown_seconds", 60.0))
    reload_on_request = bool(routing.get("reload_on_request", True))
    load_aware = bool(routing.get("load_aware", False))
    load_probe_interval_seconds = float(
        routing.get("load_probe_interval_seconds", 5.0)
    )
    load_probe_timeout_seconds = float(
        routing.get("load_probe_timeout_seconds", 1.0)
    )
    max_waiting_requests = int(routing.get("max_waiting_requests", 0))
    if failure_threshold < 1:
        raise ValueError("endpoint pool failure_threshold must be >= 1")
    if cooldown_seconds < 0:
        raise ValueError("endpoint pool cooldown_seconds must be >= 0")
    if load_probe_interval_seconds < 0:
        raise ValueError(
            "endpoint pool load_probe_interval_seconds must be >= 0"
        )
    if load_probe_timeout_seconds <= 0:
        raise ValueError(
            "endpoint pool load_probe_timeout_seconds must be > 0"
        )
    if max_waiting_requests < 0:
        raise ValueError("endpoint pool max_waiting_requests must be >= 0")
    api = raw.get("api") or {}
    if not isinstance(api, dict):
        raise ValueError(f"endpoint pool api must be a mapping: {config_path}")
    wire_api = str(api.get("wire_api") or "chat_completions").strip().lower()
    wire_api = {
        "chat": "chat_completions",
        "chat.completions": "chat_completions",
        "chat-completions": "chat_completions",
        "response": "responses",
    }.get(wire_api, wire_api)
    if wire_api not in {"chat_completions", "responses"}:
        raise ValueError(
            "endpoint pool api.wire_api must be chat_completions or responses"
        )
    api_key_env = str(api.get("api_key_env") or "").strip() or None
    reasoning_effort = str(api.get("reasoning_effort") or "").strip() or None
    model = str(raw.get("model") or "").strip() or None
    return EndpointPoolConfig(
        path=config_path,
        model=model,
        base_urls=base_urls,
        wire_api=wire_api,
        api_key_env=api_key_env,
        reasoning_effort=reasoning_effort,
        store=bool(api.get("store", False)),
        trust_env=bool(api.get("trust_env", True)),
        reload_on_request=reload_on_request,
        failure_threshold=failure_threshold,
        cooldown_seconds=cooldown_seconds,
        load_aware=load_aware,
        load_probe_interval_seconds=load_probe_interval_seconds,
        load_probe_timeout_seconds=load_probe_timeout_seconds,
        max_waiting_requests=max_waiting_requests,
    )


class DynamicEndpointPool:
    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        base_urls: Iterable[str] | str | None = None,
        model: str | None = None,
        failure_threshold: int = 1,
        cooldown_seconds: float = 60.0,
        load_aware: bool = False,
        load_probe_interval_seconds: float = 5.0,
        load_probe_timeout_seconds: float = 1.0,
        max_waiting_requests: int = 0,
    ) -> None:
        self.config_path = (
            Path(config_path).expanduser().resolve() if config_path else None
        )
        self._lock = threading.Lock()
        self._load_probe_lock = threading.Lock()
        self._next_index = 0
        self._signature: tuple[int, int] | None = None
        self._failed_signature: tuple[int, int] | None = None
        self._health: dict[str, _EndpointHealth] = {}
        self._model = str(model or "").strip() or None
        self._base_urls = normalize_base_urls(base_urls)
        self._failure_threshold = int(failure_threshold)
        self._cooldown_seconds = float(cooldown_seconds)
        self._load_aware = bool(load_aware)
        self._load_probe_interval_seconds = float(load_probe_interval_seconds)
        self._load_probe_timeout_seconds = float(load_probe_timeout_seconds)
        self._max_waiting_requests = int(max_waiting_requests)
        self._reload_on_request = True
        if self.config_path:
            self._reload(force=True)
        elif not self._base_urls:
            raise ValueError("endpoint pool requires config_path or base_urls")
        else:
            self._health = {url: _EndpointHealth() for url in self._base_urls}

    @property
    def model(self) -> str | None:
        self.refresh()
        with self._lock:
            return self._model

    @property
    def base_urls(self) -> tuple[str, ...]:
        self.refresh()
        with self._lock:
            return self._base_urls

    def refresh(self) -> bool:
        if not self.config_path or not self._reload_on_request:
            return False
        return self._reload(force=False)

    def acquire(self, *, exclude: Iterable[str] = ()) -> str:
        self.refresh()
        excluded = set(exclude)
        now = time.monotonic()
        with self._lock:
            if not self._base_urls:
                raise RuntimeError("endpoint pool has no configured base URLs")
            count = len(self._base_urls)
            ordered = [
                self._base_urls[(self._next_index + offset) % count]
                for offset in range(count)
            ]
            available = [
                url
                for url in ordered
                if url not in excluded
                and self._health[url].cooldown_until <= now
            ]
            if not available:
                available = [url for url in ordered if url not in excluded]
            if not available:
                available = ordered
            load_aware = self._load_aware

        if load_aware:
            self._refresh_endpoint_loads(available)

        with self._lock:
            available = [url for url in available if url in self._health]
            known_loads = [
                url
                for url in available
                if self._health[url].load is not None
                and self._health[url].load_error is None
            ]
            preferred = [
                url
                for url in known_loads
                if self._health[url].load.waiting_requests
                <= self._max_waiting_requests
            ]
            load_candidates = preferred or known_loads
            if load_aware and load_candidates:
                order = {url: index for index, url in enumerate(available)}
                selected = min(
                    load_candidates,
                    key=lambda url: self._load_sort_key(url, order[url]),
                )
            else:
                selected = min(
                    available,
                    key=lambda url: self._health[url].cooldown_until,
                )
            self._next_index = (self._base_urls.index(selected) + 1) % count
            return selected

    def retry_after(self, *, exclude: Iterable[str] = ()) -> float:
        self.refresh()
        excluded = set(exclude)
        now = time.monotonic()
        with self._lock:
            candidates = [url for url in self._base_urls if url not in excluded]
            if not candidates:
                candidates = list(self._base_urls)
            if not candidates:
                return 0.0
            return min(
                max(0.0, self._health[url].cooldown_until - now)
                for url in candidates
            )

    def _load_sort_key(self, base_url: str, order: int) -> tuple[float, ...]:
        load = self._health[base_url].load
        if load is None:
            return (float("inf"), float("inf"), float("inf"), float("inf"), order)
        return (
            load.waiting_requests,
            load.utilization,
            load.running_requests,
            load.pending_tokens,
            order,
        )

    def _refresh_endpoint_loads(self, base_urls: Iterable[str]) -> None:
        with self._load_probe_lock:
            now = time.monotonic()
            with self._lock:
                stale = [
                    url
                    for url in base_urls
                    if url in self._health
                    and now - self._health[url].load_checked_at
                    >= self._load_probe_interval_seconds
                ]
            if not stale:
                return
            results: dict[str, EndpointLoad | Exception] = {}
            with ThreadPoolExecutor(max_workers=min(32, len(stale))) as executor:
                futures = {
                    executor.submit(self._probe_load, url): url for url in stale
                }
                for future in as_completed(futures):
                    url = futures[future]
                    try:
                        results[url] = future.result()
                    except Exception as exc:
                        results[url] = exc
            checked_at = time.monotonic()
            with self._lock:
                for url, result in results.items():
                    health = self._health.get(url)
                    if health is None:
                        continue
                    health.load_checked_at = checked_at
                    if isinstance(result, Exception):
                        health.load = None
                        health.load_error = str(result)
                    else:
                        health.load = result
                        health.load_error = None

    def _probe_load(self, base_url: str) -> EndpointLoad:
        return probe_endpoint_load(
            base_url,
            timeout_seconds=self._load_probe_timeout_seconds,
        )

    def mark_success(self, base_url: str) -> None:
        with self._lock:
            health = self._health.get(base_url)
            if health is None:
                return
            health.consecutive_failures = 0
            health.cooldown_until = 0.0
            health.last_error = None

    def mark_failure(self, base_url: str, error: BaseException | str) -> None:
        with self._lock:
            health = self._health.get(base_url)
            if health is None:
                return
            health.consecutive_failures += 1
            health.last_error = str(error)
            if health.consecutive_failures >= self._failure_threshold:
                health.cooldown_until = time.monotonic() + self._cooldown_seconds

    def errors(self, urls: Iterable[str] | None = None) -> dict[str, str]:
        selected = set(urls) if urls is not None else None
        with self._lock:
            return {
                url: health.last_error
                for url, health in self._health.items()
                if health.last_error and (selected is None or url in selected)
            }

    def _reload(self, *, force: bool) -> bool:
        if self.config_path is None:
            return False
        try:
            stat = self.config_path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            if self._base_urls:
                logger.warning(
                    "endpoint pool file unavailable; keeping last valid config: %s",
                    self.config_path,
                )
                return False
            raise
        with self._lock:
            if not force and signature == self._signature:
                return False
            if not force and signature == self._failed_signature:
                return False
        try:
            config = load_endpoint_pool_config(self.config_path)
        except Exception:
            if self._base_urls:
                with self._lock:
                    self._failed_signature = signature
                logger.warning(
                    "invalid endpoint pool update; keeping last valid config: %s",
                    self.config_path,
                    exc_info=True,
                )
                return False
            raise
        with self._lock:
            old_health = self._health
            self._model = config.model or self._model
            self._base_urls = config.base_urls
            self._failure_threshold = config.failure_threshold
            self._cooldown_seconds = config.cooldown_seconds
            self._reload_on_request = config.reload_on_request
            self._load_aware = config.load_aware
            self._load_probe_interval_seconds = config.load_probe_interval_seconds
            self._load_probe_timeout_seconds = config.load_probe_timeout_seconds
            self._max_waiting_requests = config.max_waiting_requests
            self._health = {
                url: old_health.get(url, _EndpointHealth()) for url in self._base_urls
            }
            self._next_index %= len(self._base_urls)
            self._signature = signature
            self._failed_signature = None
        return True


def probe_endpoint_load(
    base_url: str,
    *,
    timeout_seconds: float = 1.0,
) -> EndpointLoad:
    last_error: Exception | None = None
    for load_url in _load_probe_urls(base_url):
        request = Request(load_url, headers={"Accept": "application/json"})
        try:
            with build_opener(ProxyHandler({})).open(
                request,
                timeout=timeout_seconds,
            ) as response:
                payload = json.load(response)
            return _parse_endpoint_load(payload)
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {404, 405}:
                raise
        except Exception as exc:
            last_error = exc
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"no load probe URL available for {base_url}")


def _load_probe_urls(base_url: str) -> tuple[str, str]:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    root_path = path[:-3] if path.endswith("/v1") else path

    def with_path(suffix: str) -> str:
        full_path = f"{root_path}/{suffix}".replace("//", "/")
        return urlunparse((parsed.scheme, parsed.netloc, full_path, "", "", ""))

    return with_path("v1/loads"), with_path("get_load")


def _parse_endpoint_load(payload: object) -> EndpointLoad:
    if isinstance(payload, dict) and isinstance(payload.get("loads"), list):
        entries = payload["loads"]
    elif isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = [payload]
    else:
        raise ValueError("endpoint load response must be a mapping or list")
    if not entries or not all(isinstance(item, dict) for item in entries):
        raise ValueError("endpoint load response contains no load records")

    running_requests = sum(
        _nonnegative_int(item.get("num_running_reqs", item.get("num_reqs", 0)))
        for item in entries
    )
    waiting_requests = sum(
        _nonnegative_int(item.get("num_waiting_reqs", 0)) for item in entries
    )
    pending_tokens = sum(
        _nonnegative_int(
            item.get("num_waiting_uncached_tokens", item.get("num_pending_tokens", 0))
        )
        for item in entries
    )
    utilization = max(float(item.get("utilization", 0.0) or 0.0) for item in entries)
    return EndpointLoad(
        running_requests=running_requests,
        waiting_requests=waiting_requests,
        pending_tokens=pending_tokens,
        utilization=utilization,
    )


def _nonnegative_int(value: object) -> int:
    result = int(value or 0)
    if result < 0:
        raise ValueError("endpoint load values must be non-negative")
    return result


def is_endpoint_failure(error: BaseException) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code is not None:
        try:
            code = int(status_code)
        except (TypeError, ValueError):
            code = 0
        return code in {408, 429} or code >= 500
    name = error.__class__.__name__.lower()
    if any(marker in name for marker in ("connection", "timeout", "ratelimit")):
        return True
    return isinstance(error, (ConnectionError, TimeoutError))


def is_local_endpoint(base_url: str) -> bool:
    hostname = (urlparse(base_url).hostname or "").lower()
    if hostname in {"localhost", "host.docker.internal"} or hostname.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local
