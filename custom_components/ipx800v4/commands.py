"""Single-attempt command transport and bounded, non-blocking write retries."""

import asyncio
import logging
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from typing import Any

from aiohttp import BasicAuth, ClientError, ClientResponseError, InvalidURL
from homeassistant.exceptions import HomeAssistantError
from pypx800 import (
    IPX800,
    Ipx800CannotConnectError,
    Ipx800InvalidAuthError,
    Ipx800RequestError,
)

_LOGGER = logging.getLogger(__name__)
RETRY_DELAYS = (1, 2)
_CURRENT_COMMAND: ContextVar[Callable[[], None]] = ContextVar("ipx_command_check")


class IpxCommandClient(IPX800):
    """Avoid pypx800 2.5.1's blocking sleep even after a single CGI failure."""

    async def request_cgi(self, params: dict) -> str:
        """Send once; keep retries at the explicitly opted-in operation level."""
        auth = None
        if self._username and self._password:
            auth = BasicAuth(self._username, self._password)
        try:
            async with asyncio.timeout(self._request_timeout):
                response = await self._session.get(
                    self._cgi_url, auth=auth, params=params
                )
                try:
                    if response.status in (401, 403):
                        raise Ipx800InvalidAuthError("IPX800 authentication failed")
                    response.raise_for_status()
                    content = await response.text()
                finally:
                    response.close()
            if not self._request_checkstatus or "Success" in content:
                return content
            raise Ipx800RequestError("IPX800 rejected the CGI request")
        except (TimeoutError, ClientError, socket.gaierror) as err:
            raise Ipx800CannotConnectError("IPX800 communication failed") from err


def transient_error(error: Exception) -> bool:
    """Do not replay rejected requests, invalid URLs or definitive HTTP errors."""
    cause = error.__cause__
    if isinstance(cause, InvalidURL):
        return False
    if isinstance(cause, ClientResponseError):
        return cause.status in (408, 429) or 500 <= cause.status < 600
    return isinstance(error, (Ipx800CannotConnectError, TimeoutError))


class CommandManager:
    """Coordinate overlapping outputs, including aliases in different platforms."""

    def __init__(self) -> None:
        self._closed = False
        self._versions: dict[str, object] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def shutdown(self) -> None:
        """Prevent pending retries or remaining writes after entry unload."""
        self._closed = True

    @asynccontextmanager
    async def operation(self, keys: tuple[str, ...]) -> AsyncIterator[None]:
        # Register intent before waiting for an in-flight write. An older
        # operation must not retry or write its remaining channels afterwards.
        version = object()
        for key in keys:
            self._versions[key] = version

        def check_current() -> None:
            if self._closed:
                raise HomeAssistantError("IPX800 integration is unloading")
            if any(self._versions[key] is not version for key in keys):
                raise HomeAssistantError("IPX800 command superseded by a newer command")

        token = _CURRENT_COMMAND.set(check_current)
        try:
            yield
        finally:
            _CURRENT_COMMAND.reset(token)

    async def write(
        self,
        keys: tuple[str, ...],
        command: Callable[..., Awaitable[Any]],
        *args: Any,
        retry: bool = False,
        **kwargs: Any,
    ) -> None:
        """Retry one fixed write; release locks during asynchronous backoff."""
        for attempt in range(len(RETRY_DELAYS) + 1 if retry else 1):
            _CURRENT_COMMAND.get()()
            try:
                async with AsyncExitStack() as stack:
                    for key in sorted(set(keys)):
                        await stack.enter_async_context(
                            self._locks.setdefault(key, asyncio.Lock())
                        )
                    _CURRENT_COMMAND.get()()
                    await command(*args, **kwargs)
                return
            except (Ipx800CannotConnectError, TimeoutError) as err:
                if (
                    not retry
                    or attempt == len(RETRY_DELAYS)
                    or not transient_error(err)
                ):
                    raise
                _CURRENT_COMMAND.get()()
                _LOGGER.debug(
                    "IPX800 write failed (%s); attempt %s/%s in %ss",
                    type(err).__name__,
                    attempt + 2,
                    len(RETRY_DELAYS) + 1,
                    RETRY_DELAYS[attempt],
                )
                await asyncio.sleep(RETRY_DELAYS[attempt])
