"""Build dataclasses from plain dicts (parsed YAML or msgpack) with strict type checks.

Used by config loading and message decoding so a typo or wrong type fails loudly at the boundary
instead of deep inside a control loop.
"""

import dataclasses
import functools
import types
import typing
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin


class SchemaError(ValueError):
    pass


_hints = functools.cache(typing.get_type_hints)


def from_dict[T](cls: type[T], data: Any, path: str = "") -> T:
    return _build(cls, data, path or cls.__name__)


def _build(tp: Any, value: Any, path: str) -> Any:
    if tp is Any:
        return value
    origin = get_origin(tp)

    if origin in (Union, types.UnionType):
        args = get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise SchemaError(f"{path}: must not be null")
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _build(arg, value, path)
            except SchemaError:
                pass
        raise SchemaError(f"{path}: {value!r} does not match {tp}")

    if dataclasses.is_dataclass(tp):
        if isinstance(value, tp):
            return value
        if not isinstance(value, dict):
            raise SchemaError(f"{path}: expected a mapping, got {type(value).__name__}")
        fields = {f.name: f for f in dataclasses.fields(tp) if f.init}
        unknown = set(value) - set(fields)
        if unknown:
            raise SchemaError(f"{path}: unknown keys {sorted(unknown)}")
        hints = _hints(tp)
        kwargs = {}
        for name, f in fields.items():
            if name in value:
                kwargs[name] = _build(hints[name], value[name], f"{path}.{name}")
            elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                raise SchemaError(f"{path}: missing required key '{name}'")
        try:
            return tp(**kwargs)
        except (TypeError, ValueError) as e:
            raise SchemaError(f"{path}: {e}") from e

    if origin is list:
        (item_tp,) = get_args(tp) or (Any,)
        if not isinstance(value, list | tuple):
            raise SchemaError(f"{path}: expected a list, got {type(value).__name__}")
        return [_build(item_tp, v, f"{path}[{i}]") for i, v in enumerate(value)]

    if origin is dict:
        key_tp, val_tp = get_args(tp) or (Any, Any)
        if not isinstance(value, dict):
            raise SchemaError(f"{path}: expected a mapping, got {type(value).__name__}")
        return {
            _build(key_tp, k, f"{path}.<key>"): _build(val_tp, v, f"{path}.{k}")
            for k, v in value.items()
        }

    if origin is Literal:
        if value not in get_args(tp):
            raise SchemaError(f"{path}: {value!r} is not one of {get_args(tp)}")
        return value

    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SchemaError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaError(f"{path}: expected an integer, got {value!r}")
        return value
    if tp is Path:
        if not isinstance(value, str | Path):
            raise SchemaError(f"{path}: expected a path, got {value!r}")
        return Path(value)
    if tp in (str, bool, bytes):
        if not isinstance(value, tp):
            raise SchemaError(f"{path}: expected {tp.__name__}, got {value!r}")
        return value

    raise SchemaError(f"{path}: unsupported type {tp}")
