"""Unified benchmark-loop substrate.

Every runner in the kit (CLI harness benchmarks, IFEval+, bundled-bench)
reduces to the same shape:

  * iterate input items
  * format each into a prompt (and some metadata the record builder needs)
  * dispatch to a *responder* that returns ``(response_text, latency_ms)``
  * build a record from ``(idx, item, prompt, meta, response, latency)``
  * stream the record to an archive

This module gives that structure a home. Two entry points:

  * :func:`run_items` — synchronous for CLI-harness transports.
  * :func:`run_items_async` — async with bounded concurrency for HTTP
    transports. Failures become explicit failure records so cell n-counts
    stay honest.

Responders are produced by the factories in :mod:`.responders`.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    TypeVar,
    Union,
)

from .archive import ArchiveWriter

__all__ = [
    "AsyncResponder",
    "BuildFailureRecord",
    "BuildRecord",
    "FormatFn",
    "ProgressFn",
    "Serialize",
    "SyncResponder",
    "run_items",
    "run_items_async",
]

T = TypeVar("T")  # input item type
U = TypeVar("U")  # metadata produced by format_fn, consumed by build_record
R = TypeVar("R")  # output record type (typically a dataclass or TypedDict)

SyncResponder = Callable[[str, Union[str, None]], "tuple[str, int]"]
AsyncResponder = Callable[[str, Union[str, None]], Awaitable["tuple[str, int]"]]
FormatFn = Callable[[T], "tuple[str, U]"]
BuildRecord = Callable[[int, T, str, U, str, int], R]
BuildFailureRecord = Callable[[int, T, str, U, BaseException], R]
Serialize = Callable[[R], Mapping[str, Any]]
ProgressFn = Callable[[int, R], None]


def _default_serialize(record: Any) -> Mapping[str, Any]:
    """Turn a record into a JSON-friendly mapping.

    Supports dataclasses (via ``asdict``) and plain mappings (passed
    through). Anything else is rejected loudly — silent "str(record)" is
    worse than a crash.
    """
    if is_dataclass(record) and not isinstance(record, type):
        return asdict(record)
    if isinstance(record, Mapping):
        return dict(record)  # type: ignore[arg-type]
    raise TypeError(
        f"run_items: record of type {type(record).__name__} is not a dataclass "
        "or Mapping; pass a custom `serialize` callable"
    )


def _open_writer(
    output_dir: str | Path | None, description: str
) -> ArchiveWriter | None:
    """Open an ArchiveWriter if output is wanted, else return None.

    Caller is responsible for close/finalize — a plain with-block around the
    returned writer suffices. The reason we do NOT use a @contextmanager
    helper here is that ``final_path`` is only populated in
    ``ArchiveWriter.__exit__``; yielding the writer inside a generator-based
    context manager means the outer block reads ``final_path`` BEFORE the
    inner __exit__ has run.
    """
    if output_dir is None:
        return None
    return ArchiveWriter(Path(output_dir), description)


def _apply_limit(items: Iterable[T], limit: int | None) -> Iterator[T]:
    for idx, item in enumerate(items):
        if limit is not None and idx >= limit:
            return
        yield item


class _BatchState(Generic[R]):
    """Shared sink state for a run. Lets us keep returns typed per call."""

    def __init__(
        self,
        writer: ArchiveWriter | None,
        serialize: Serialize[R],
        on_progress: ProgressFn[R] | None,
    ) -> None:
        self.records: list[R] = []
        self._writer = writer
        self._serialize = serialize
        self._on_progress = on_progress

    def emit(self, idx: int, record: R) -> None:
        self.records.append(record)
        if self._writer is not None:
            self._writer.write_record(self._serialize(record))
        if self._on_progress is not None:
            self._on_progress(idx, record)


def run_items(
    items: Iterable[T],
    responder: SyncResponder,
    format_fn: FormatFn[T, U],
    build_record: BuildRecord[T, U, R],
    *,
    description: str,
    output_dir: str | Path | None = None,
    system_prompt: str | None = None,
    limit: int | None = None,
    serialize: Serialize[R] = _default_serialize,
    on_progress: ProgressFn[R] | None = None,
) -> tuple[list[R], Path | None]:
    """Run a synchronous per-item benchmark loop.

    Streams each record to the archive the moment it's built (so a crash
    at item 99 preserves 0–98). Exceptions from the responder or
    build_record propagate — callers who need failure-record discipline
    should catch inside their build_record or use :func:`run_items_async`.
    """
    writer = _open_writer(output_dir, description)
    state: _BatchState[R]
    if writer is None:
        state = _BatchState[R](None, serialize, on_progress)
        for idx, item in enumerate(_apply_limit(items, limit)):
            prompt, meta = format_fn(item)
            response, latency_ms = responder(prompt, system_prompt)
            record = build_record(idx, item, prompt, meta, response, latency_ms)
            state.emit(idx, record)
        return state.records, None
    with writer as w:
        state = _BatchState[R](w, serialize, on_progress)
        for idx, item in enumerate(_apply_limit(items, limit)):
            prompt, meta = format_fn(item)
            response, latency_ms = responder(prompt, system_prompt)
            record = build_record(idx, item, prompt, meta, response, latency_ms)
            state.emit(idx, record)
    # final_path is populated in ArchiveWriter.__exit__, read after block.
    return state.records, writer.final_path


async def run_items_async(
    items: Iterable[T],
    responder: AsyncResponder,
    format_fn: FormatFn[T, U],
    build_record: BuildRecord[T, U, R],
    *,
    description: str,
    output_dir: str | Path | None = None,
    system_prompt: str | None = None,
    limit: int | None = None,
    max_concurrent: int = 5,
    serialize: Serialize[R] = _default_serialize,
    on_progress: ProgressFn[R] | None = None,
    build_failure_record: BuildFailureRecord[T, U, R] | None = None,
) -> tuple[list[R], Path | None]:
    """Run an async benchmark loop with bounded concurrency.

    All items run concurrently (bounded by ``max_concurrent``); results
    are written in input order to keep archives deterministic. Failures
    become explicit records via ``build_failure_record`` so cell n-counts
    stay honest. If ``build_failure_record`` is None, exceptions are
    re-raised (strict mode).
    """
    items_list = list(_apply_limit(items, limit))
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(idx: int, item: T) -> tuple[int, T, str, U, str, int, BaseException | None]:
        prompt, meta = format_fn(item)
        async with sem:
            try:
                response, ms = await responder(prompt, system_prompt)
            except BaseException as e:  # noqa: BLE001 — we re-surface or record
                return idx, item, prompt, meta, "", 0, e
        return idx, item, prompt, meta, response, ms, None

    tasks = [_one(i, it) for i, it in enumerate(items_list)]
    raw = await asyncio.gather(*tasks)

    writer = _open_writer(output_dir, description)
    state: _BatchState[R]

    def _process_one(idx: int, item: T, prompt: str, meta: U, resp: str, ms: int, exc: BaseException | None) -> R:
        if exc is not None:
            if build_failure_record is None:
                raise exc
            print(f"ERROR idx={idx}: {exc}", file=sys.stderr)
            return build_failure_record(idx, item, prompt, meta, exc)
        return build_record(idx, item, prompt, meta, resp, ms)

    if writer is None:
        state = _BatchState[R](None, serialize, on_progress)
        for idx, item, prompt, meta, resp, ms, exc in raw:
            record = _process_one(idx, item, prompt, meta, resp, ms, exc)
            state.emit(idx, record)
        return state.records, None
    with writer as w:
        state = _BatchState[R](w, serialize, on_progress)
        for idx, item, prompt, meta, resp, ms, exc in raw:
            record = _process_one(idx, item, prompt, meta, resp, ms, exc)
            state.emit(idx, record)
    return state.records, writer.final_path
