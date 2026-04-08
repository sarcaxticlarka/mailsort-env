"""
MailSort environment client.

Provides MailSortEnv — an async client that connects to a running
MailSort server (local, Docker, or HuggingFace Space) and exposes
the standard OpenEnv interface:

    reset(**kwargs)  -> StepResult
    step(action)     -> StepResult
    state()          -> MailSortState
    close()          -> None

Usage (async):
    async with MailSortEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task="email_classify")
        action = MailSortAction(classifications=[...])
        result = await env.step(action)

Usage (sync):
    with MailSortEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset(task="email_rank")

Usage (Docker):
    env = MailSortEnv.from_docker_image("mailsort-env:latest")
    try:
        result = await env.reset(task="email_triage")
    finally:
        await env.close()

Usage (HuggingFace Space):
    env = MailSortEnv.from_env("your-hf-username/mailsort-env")
    result = await env.reset(task="email_triage")
"""

from __future__ import annotations

from models import MailSortAction, MailSortObservation, MailSortState

try:
    from openenv.core.env_client import EnvClient

    class MailSortEnv(EnvClient[MailSortAction, MailSortObservation, MailSortState]):
        """
        Client for the MailSort environment.

        All functionality is inherited from EnvClient:
          - reset(**kwargs)         -> StepResult[MailSortObservation]
          - step(action)            -> StepResult[MailSortObservation]
          - state()                 -> MailSortState
          - connect()               -> MailSortEnv
          - close()                 -> None
          - sync()                  -> SyncEnvClient
          - from_docker_image(...)  -> MailSortEnv  (class method)
          - from_env(repo_id, ...)  -> MailSortEnv  (class method)

        Task selection is passed through reset():
            await env.reset(task="email_classify")   # easy
            await env.reset(task="email_rank")        # medium
            await env.reset(task="email_triage")      # hard
        """
        pass

except ImportError:
    # ---------------------------------------------------------------------------
    # Fallback HTTP client — used when openenv-core is not installed.
    # Provides the same async interface backed by httpx HTTP calls.
    # ---------------------------------------------------------------------------

    import asyncio
    import subprocess
    import time
    from dataclasses import dataclass
    from typing import Any, Dict, Generic, Optional, TypeVar

    try:
        import httpx
        _HAS_HTTPX = True
    except ImportError:
        _HAS_HTTPX = False
        import urllib.request
        import json as _json

    @dataclass
    class StepResult:
        """Mirrors openenv-core StepResult."""
        observation: MailSortObservation
        reward: float
        done: bool
        info: Dict[str, Any] = None

        def __post_init__(self):
            if self.info is None:
                self.info = {}

    class _SyncWrapper:
        """Synchronous wrapper over the async MailSortEnv."""
        def __init__(self, env: "MailSortEnv"):
            self._env = env
            self._loop = asyncio.new_event_loop()

        def __enter__(self):
            self._loop.run_until_complete(self._env.connect())
            return self

        def __exit__(self, *_):
            self._loop.run_until_complete(self._env.close())
            self._loop.close()

        def reset(self, **kwargs) -> StepResult:
            return self._loop.run_until_complete(self._env.reset(**kwargs))

        def step(self, action: MailSortAction) -> StepResult:
            return self._loop.run_until_complete(self._env.step(action))

        def state(self) -> MailSortState:
            return self._loop.run_until_complete(self._env.state())

    class MailSortEnv:
        """Fallback async HTTP client for MailSort environment."""

        def __init__(self, base_url: str = "http://localhost:8000") -> None:
            self.base_url = base_url.rstrip("/")
            self._client: Optional[Any] = None
            self._docker_proc: Optional[subprocess.Popen] = None

        # --- Context manager ---

        async def __aenter__(self) -> "MailSortEnv":
            await self.connect()
            return self

        async def __aexit__(self, *_) -> None:
            await self.close()

        async def connect(self) -> "MailSortEnv":
            if _HAS_HTTPX:
                self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
            return self

        async def close(self) -> None:
            if self._client and _HAS_HTTPX:
                await self._client.aclose()
            if self._docker_proc:
                self._docker_proc.terminate()
                try:
                    self._docker_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._docker_proc.kill()

        def sync(self) -> _SyncWrapper:
            return _SyncWrapper(self)

        # --- Core methods ---

        async def reset(self, **kwargs) -> StepResult:
            payload = {k: v for k, v in kwargs.items() if v is not None}
            data = await self._post("/reset", payload)
            obs = self._parse_obs(data.get("observation", {}))
            return StepResult(
                observation=obs,
                reward=float(data.get("reward", 0.0)),
                done=bool(data.get("done", False)),
            )

        async def step(self, action: MailSortAction) -> StepResult:
            if hasattr(action, "model_dump"):
                action_dict = action.model_dump(exclude_none=True)
            else:
                action_dict = {k: v for k, v in action.dict().items() if v is not None}
            action_dict.pop("metadata", None)
            # openenv-core create_app wraps action in {"action": {...}}
            payload = {"action": action_dict}
            data = await self._post("/step", payload)
            obs = self._parse_obs(data.get("observation", {}))
            return StepResult(
                observation=obs,
                reward=float(data.get("reward", 0.0)),
                done=bool(data.get("done", False)),
            )

        async def state(self) -> MailSortState:
            data = await self._get("/state")
            return MailSortState(**data)

        # --- Factory class methods ---

        @classmethod
        def from_docker_image(
            cls,
            image: str,
            port: int = 7860,
            startup_timeout: int = 60,
            **kwargs,
        ) -> "MailSortEnv":
            """Start a Docker container from `image` and return a connected client."""
            proc = subprocess.Popen(
                ["docker", "run", "--rm", "-p", f"{port}:7860", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Wait for server to become ready
            deadline = time.time() + startup_timeout
            while time.time() < deadline:
                try:
                    if _HAS_HTTPX:
                        import httpx as _httpx
                        r = _httpx.get(f"http://localhost:{port}/health", timeout=2)
                        if r.status_code == 200:
                            break
                    else:
                        urllib.request.urlopen(
                            f"http://localhost:{port}/health", timeout=2
                        )
                        break
                except Exception:
                    time.sleep(1)
            else:
                proc.terminate()
                raise RuntimeError(
                    f"Docker container {image!r} did not become ready within {startup_timeout}s"
                )
            env = cls(base_url=f"http://localhost:{port}")
            env._docker_proc = proc
            return env

        @classmethod
        def from_env(cls, repo_id: str, **kwargs) -> "MailSortEnv":
            """Connect to a MailSort environment running on HuggingFace Spaces."""
            # HF Space URLs follow the pattern: https://<user>-<space>.hf.space
            slug = repo_id.replace("/", "-").lower()
            base_url = f"https://{slug}.hf.space"
            return cls(base_url=base_url)

        # --- HTTP helpers ---

        async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            if _HAS_HTTPX:
                if self._client is None:
                    await self.connect()
                r = await self._client.post(path, json=payload)
                r.raise_for_status()
                return r.json()
            else:
                return self._urllib_post(path, payload)

        async def _get(self, path: str) -> Dict[str, Any]:
            if _HAS_HTTPX:
                if self._client is None:
                    await self.connect()
                r = await self._client.get(path)
                r.raise_for_status()
                return r.json()
            else:
                return self._urllib_get(path)

        def _urllib_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            import json as _json
            import urllib.request
            data = _json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.base_url}{path}",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return _json.loads(resp.read())

        def _urllib_get(self, path: str) -> Dict[str, Any]:
            import json as _json
            import urllib.request
            with urllib.request.urlopen(f"{self.base_url}{path}", timeout=30) as resp:
                return _json.loads(resp.read())

        @staticmethod
        def _parse_obs(d: Dict[str, Any]) -> MailSortObservation:
            # Remove keys unknown to the model
            known = {f for f in MailSortObservation.model_fields} if hasattr(
                MailSortObservation, "model_fields"
            ) else set()
            if known:
                d = {k: v for k, v in d.items() if k in known}
            try:
                return MailSortObservation(**d)
            except Exception:
                return MailSortObservation(
                    task_name=d.get("task_name", ""),
                    task_description=d.get("task_description", ""),
                    step_description=d.get("step_description", ""),
                    emails=d.get("emails", []),
                    step=d.get("step", 0),
                    max_steps=d.get("max_steps", 1),
                    feedback=d.get("feedback"),
                    last_action_error=d.get("last_action_error"),
                    done=d.get("done", False),
                    reward=d.get("reward", 0.0),
                )
