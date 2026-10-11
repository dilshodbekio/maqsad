"""Request-size limit as a pure ASGI middleware.

It counts the bytes that were really received (the Content-Length header is only a hint: a client can lie or send chunked),
so an oversized body is cut off before the multipart parser or the JSON parser has buffered it all.
Limits: the photo endpoint `photo_limit` bytes, every other POST/PUT/PATCH/DELETE `other_limit` (64 KB). Over the limit -> 413.
"""
from __future__ import annotations

import json
from typing import Callable

from . import i18n
from .i18n import Msg

OTHER_BODY_LIMIT = 64 * 1024
BODY_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
MSG_BODY_TOO_LARGE = Msg("body_too_large", kb=OTHER_BODY_LIMIT // 1024)


class BodyLimitMiddleware:
    def __init__(self, app, *, limit_for: Callable[[str], tuple[int, Msg]]):
        """`limit_for(path)` -> (max body bytes, the Msg for the 413 text)."""
        self.app, self.limit_for = app, limit_for

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in BODY_METHODS:
            await self.app(scope, receive, send)
            return
        limit, msg = self.limit_for(scope["path"])
        headers = {k: v for k, v in scope["headers"]}
        lang = i18n.parse_lang(headers.get(b"accept-language", b"").decode("latin-1"))
        declared = headers.get(b"content-length", b"")
        if declared.isdigit() and int(declared) > limit:
            await self._reply_413(send, msg, lang)
            return

        received, exceeded, started = 0, False, False

        async def limited_receive():
            nonlocal received, exceeded
            if exceeded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True  # stop feeding the app; whatever it answers is replaced by the 413 below
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            nonlocal started
            if exceeded:
                return  # the app saw a "disconnect"; its reply is dropped
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded and not started:
            await self._reply_413(send, msg, lang)

    @staticmethod
    async def _reply_413(send, msg: Msg, lang: str) -> None:
        body = json.dumps({"detail": msg.render(lang)}, ensure_ascii=False).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                                (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": body})
