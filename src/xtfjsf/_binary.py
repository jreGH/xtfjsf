"""Small helper for declaring fixed-layout little-endian binary records."""

from __future__ import annotations

import re
import struct
from typing import Any, Dict, Iterable, List, Sequence, Tuple

_COUNT_RE = re.compile(r"^(\d*)([a-zA-Z?])$")


class BinaryRecord:
    """A named, fixed-size binary layout.

    ``fields`` is a sequence of ``(name, fmt)`` pairs where ``fmt`` is a single
    :mod:`struct` code optionally prefixed with a repeat count (``"6f"``,
    ``"16s"``).  Fields whose name starts with ``_`` are treated as padding:
    they are skipped when unpacking and zero-filled when packing.
    """

    def __init__(self, name: str, fields: Sequence[Tuple[str, str]], size: int | None = None):
        self.name = name
        self.fields: List[Tuple[str, str, int]] = []
        fmt = "<"
        for fname, code in fields:
            m = _COUNT_RE.match(code)
            if not m:
                raise ValueError(f"bad format code {code!r} for {name}.{fname}")
            count = int(m.group(1) or 1)
            kind = m.group(2)
            nvalues = 1 if kind in "sx" else count
            self.fields.append((fname, code, nvalues))
            fmt += code
        self._struct = struct.Struct(fmt)
        self.size = self._struct.size
        if size is not None and size != self.size:
            raise AssertionError(f"{name}: layout is {self.size} bytes, expected {size}")

    def unpack(self, buf: bytes | memoryview, offset: int = 0) -> Dict[str, Any]:
        values = self._struct.unpack_from(buf, offset)
        out: Dict[str, Any] = {}
        i = 0
        for fname, code, n in self.fields:
            if n == 1:
                v = values[i]
                if code.endswith("s"):
                    v = _decode_str(v)
            else:
                v = values[i : i + n]
            i += n
            if not fname.startswith("_"):
                out[fname] = v
        return out

    def pack(self, values: Dict[str, Any] | None = None, **kwargs: Any) -> bytes:
        merged = dict(values or {})
        merged.update(kwargs)
        flat: List[Any] = []
        for fname, code, n in self.fields:
            v = merged.get(fname) if not fname.startswith("_") else None
            if code.endswith("s"):
                if v is None:
                    v = b""
                elif isinstance(v, str):
                    v = v.encode("latin-1", errors="replace")
                flat.append(v)
            elif n == 1:
                flat.append(0 if v is None else v)
            else:
                seq = list(v) if v is not None else []
                seq += [0] * (n - len(seq))
                flat.extend(seq[:n])
        return self._struct.pack(*flat)

    def defaults(self) -> Dict[str, Any]:
        return self.unpack(b"\x00" * self.size)


def _decode_str(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("latin-1").strip()


def names(record: BinaryRecord) -> Iterable[str]:
    return (f for f, _, _ in record.fields if not f.startswith("_"))
