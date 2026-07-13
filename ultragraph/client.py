"""Async client for ultra-graph model inference.

``Client`` provides ``async with`` / ``await`` interfaces for generating
text from a model. Designed for MCP servers, web backends, and streaming
applications.

Usage:
    async with ug.Client(model) as client:
        result = await client.generate("Hello", n_new=50)
        async for token in client.stream("Once upon"):
            print(token)
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator


class Client:
    """Async client wrapping a GPT model for streaming generation.

    Example:
        model = GPT(256, 128, 4, 4)
        async with Client(model) as client:
            tokens = await client.generate("Hello world", n_new=32)
    """

    def __init__(self, model, tokenizer=None):
        self.model = model
        self._tokenizer = tokenizer

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def generate(self, prompt, n_new: int = 32, **kwargs) -> list[int]:
        """Generate token IDs. ``prompt`` can be str (auto-tokenized) or int list.

        Returns list of int token IDs (prompt + generated).
        """
        ids = self._prepare_prompt(prompt)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.model.generate, ids, n_new, **{k: v for k, v in kwargs.items() if k != "stream"}
        )

    async def stream(self, prompt, n_new: int = 32, **kwargs) -> AsyncIterator[int]:
        """Stream generated token IDs one at a time.

        Example:
            async for token in client.stream("Hello", n_new=50):
                print(chr(token))
        """
        ids = self._prepare_prompt(prompt)
        kwargs["stream"] = True
        loop = asyncio.get_running_loop()

        def _run():
            return list(self.model.generate(ids, n_new, **kwargs))

        tokens = await loop.run_in_executor(None, _run)
        for t in tokens:
            yield t
            await asyncio.sleep(0)

    async def generate_text(self, prompt: str, n_new: int = 32, **kwargs) -> str:
        """Generate and decode to a string.

        Requires a ByteTokenizer. If none was passed, creates one with vocab=256.
        """
        tokens = await self.generate(prompt, n_new, **kwargs)
        if self._tokenizer:
            return self._tokenizer.decode(tokens)
        from .tokenize import ByteTokenizer
        tok = ByteTokenizer()
        return tok.decode(tokens)

    def _prepare_prompt(self, prompt) -> list[int]:
        if isinstance(prompt, str):
            if self._tokenizer:
                return self._tokenizer.encode(prompt)
            from .tokenize import ByteTokenizer
            tok = ByteTokenizer()
            return tok.encode(prompt)
        return list(prompt)
