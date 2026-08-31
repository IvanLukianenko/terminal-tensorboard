"""Zero-dependency reader/writer for TensorBoard event files (tfevents).

Implements just enough of the TFRecord framing and the protobuf wire format
to extract scalar summaries quickly, without TensorFlow or protobuf
installed.  Both the classic ``simple_value`` scalars (TF1 / most third-party
writers) and the TF2 tensor-encoded scalars are supported.

The parser is deliberately allocation-light: it walks a ``bytes`` buffer with
integer offsets, skips unknown fields without decoding them, and interns tag
strings so repeated tags share one object.
"""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

_U64 = struct.Struct("<Q")
_U32 = struct.Struct("<I")
_F64 = struct.Struct("<d")
_F32 = struct.Struct("<f")

# TensorProto dtypes we know how to read a scalar out of.
_DT_FLOAT = 1
_DT_DOUBLE = 2
_DT_INT32 = 3
_DT_INT64 = 9

ScalarPoint = Tuple[str, int, float, float]  # (tag, step, wall_time, value)


# --------------------------------------------------------------------------
# protobuf wire-format primitives
# --------------------------------------------------------------------------

def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if b < 0x80:
            return result, pos
        shift += 7
        if shift > 70:
            raise ValueError("varint too long")


def _skip_field(buf: bytes, pos: int, wire_type: int) -> int:
    if wire_type == 0:  # varint
        while buf[pos] >= 0x80:
            pos += 1
        return pos + 1
    if wire_type == 1:  # fixed64
        return pos + 8
    if wire_type == 2:  # length-delimited
        ln, pos = _read_varint(buf, pos)
        return pos + ln
    if wire_type == 5:  # fixed32
        return pos + 4
    raise ValueError("unsupported wire type %d" % wire_type)


def _signed64(v: int) -> int:
    return v - (1 << 64) if v >= (1 << 63) else v


# --------------------------------------------------------------------------
# TensorProto -> first scalar value
# --------------------------------------------------------------------------

def _tensor_scalar(buf: bytes, pos: int, end: int) -> Optional[float]:
    dtype = 0
    content: Optional[Tuple[int, int]] = None
    value: Optional[float] = None
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field, wt = key >> 3, key & 7
        if field == 1 and wt == 0:  # dtype
            dtype, pos = _read_varint(buf, pos)
        elif field == 4 and wt == 2:  # tensor_content
            ln, pos = _read_varint(buf, pos)
            content = (pos, pos + ln)
            pos += ln
        elif field == 5:  # float_val
            if wt == 2:  # packed
                ln, pos = _read_varint(buf, pos)
                if value is None and ln >= 4:
                    value = _F32.unpack_from(buf, pos)[0]
                pos += ln
            elif wt == 5:
                if value is None:
                    value = _F32.unpack_from(buf, pos)[0]
                pos += 4
            else:
                pos = _skip_field(buf, pos, wt)
        elif field == 6:  # double_val
            if wt == 2:
                ln, pos = _read_varint(buf, pos)
                if value is None and ln >= 8:
                    value = _F64.unpack_from(buf, pos)[0]
                pos += ln
            elif wt == 1:
                if value is None:
                    value = _F64.unpack_from(buf, pos)[0]
                pos += 8
            else:
                pos = _skip_field(buf, pos, wt)
        elif field in (7, 10):  # int_val / int64_val
            if wt == 2:
                ln, pos = _read_varint(buf, pos)
                if value is None and ln > 0:
                    v, _ = _read_varint(buf, pos)
                    value = float(_signed64(v))
                pos += ln
            elif wt == 0:
                v, pos = _read_varint(buf, pos)
                if value is None:
                    value = float(_signed64(v))
            else:
                pos = _skip_field(buf, pos, wt)
        else:
            pos = _skip_field(buf, pos, wt)

    if value is not None:
        return value
    if content is not None:
        a, b = content
        if dtype == _DT_FLOAT and b - a >= 4:
            return _F32.unpack_from(buf, a)[0]
        if dtype == _DT_DOUBLE and b - a >= 8:
            return _F64.unpack_from(buf, a)[0]
        if dtype == _DT_INT32 and b - a >= 4:
            return float(struct.unpack_from("<i", buf, a)[0])
        if dtype == _DT_INT64 and b - a >= 8:
            return float(struct.unpack_from("<q", buf, a)[0])
    return None


# --------------------------------------------------------------------------
# Summary.Value / Summary / Event
# --------------------------------------------------------------------------

def _parse_value(
    buf: bytes, pos: int, end: int, intern: Dict[bytes, str]
) -> Tuple[Optional[str], Optional[float]]:
    tag: Optional[str] = None
    value: Optional[float] = None
    while pos < end:
        key = buf[pos]  # field keys and short lengths are 1-byte varints
        if key < 0x80:
            pos += 1
        else:
            key, pos = _read_varint(buf, pos)
        if key == 0x0A:  # tag (field 1, length-delimited)
            ln = buf[pos]
            if ln < 0x80:
                pos += 1
            else:
                ln, pos = _read_varint(buf, pos)
            raw = buf[pos : pos + ln]
            tag = intern.get(raw)
            if tag is None:
                tag = raw.decode("utf-8", "replace")
                intern[raw] = tag
            pos += ln
        elif key == 0x15:  # simple_value (field 2, fixed32)
            value = _F32.unpack_from(buf, pos)[0]
            pos += 4
        elif key == 0x42:  # tensor (field 8, length-delimited)
            ln, pos = _read_varint(buf, pos)
            if value is None:
                value = _tensor_scalar(buf, pos, pos + ln)
            pos += ln
        else:
            pos = _skip_field(buf, pos, key & 7)
    return tag, value


def _parse_event(
    buf: bytes, pos: int, end: int, intern: Dict[bytes, str], out: List[ScalarPoint]
) -> None:
    wall_time = 0.0
    step = 0
    summary: Optional[Tuple[int, int]] = None
    while pos < end:
        key = buf[pos]
        if key < 0x80:
            pos += 1
        else:
            key, pos = _read_varint(buf, pos)
        if key == 0x09:  # wall_time (field 1, fixed64)
            wall_time = _F64.unpack_from(buf, pos)[0]
            pos += 8
        elif key == 0x10:  # step (field 2, varint)
            v, pos = _read_varint(buf, pos)
            step = v - (1 << 64) if v >= (1 << 63) else v
        elif key == 0x2A:  # summary (field 5, length-delimited)
            ln = buf[pos]
            if ln < 0x80:
                pos += 1
            else:
                ln, pos = _read_varint(buf, pos)
            summary = (pos, pos + ln)
            pos += ln
        else:
            pos = _skip_field(buf, pos, key & 7)

    if summary is None:
        return
    pos, end = summary
    while pos < end:  # repeated Summary.value
        key = buf[pos]
        if key < 0x80:
            pos += 1
        else:
            key, pos = _read_varint(buf, pos)
        if key == 0x0A:  # value (field 1, length-delimited)
            ln = buf[pos]
            if ln < 0x80:
                pos += 1
            else:
                ln, pos = _read_varint(buf, pos)
            tag, value = _parse_value(buf, pos, pos + ln, intern)
            if tag is not None and value is not None:
                out.append((tag, step, wall_time, value))
            pos += ln
        else:
            pos = _skip_field(buf, pos, key & 7)


# --------------------------------------------------------------------------
# TFRecord framing
# --------------------------------------------------------------------------

def parse_chunk(
    data: bytes, intern: Optional[Dict[bytes, str]] = None
) -> Tuple[List[ScalarPoint], int]:
    """Parse scalar points out of a chunk of a tfevents file.

    Returns ``(points, consumed)`` where ``consumed`` is the number of bytes
    of complete records that were processed.  A trailing partially-written
    record (a file that is still being appended to) is left unconsumed so the
    caller can retry from the same offset once more bytes exist.  A corrupt
    record aborts parsing of the chunk at that record.
    """
    if intern is None:
        intern = {}
    points: List[ScalarPoint] = []
    pos = 0
    n = len(data)
    while pos + 12 <= n:
        (length,) = _U64.unpack_from(data, pos)
        body = pos + 12
        after = body + length + 4
        if length > (1 << 31) or after > n:
            break  # truncated / still being written
        try:
            _parse_event(data, body, body + length, intern, points)
        except (ValueError, IndexError, struct.error):
            raise CorruptRecord(pos)
        pos = after
    return points, pos


class CorruptRecord(Exception):
    """Raised when a framed record contains undecodable protobuf."""

    def __init__(self, offset: int):
        super().__init__("corrupt record at offset %d" % offset)
        self.offset = offset


# --------------------------------------------------------------------------
# Writer (used by tests and the demo-log generator)
# --------------------------------------------------------------------------

_CRC_TABLE: List[int] = []


def _crc32c(data: bytes) -> int:
    if not _CRC_TABLE:
        for i in range(256):
            c = i
            for _ in range(8):
                c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
            _CRC_TABLE.append(c)
    crc = 0xFFFFFFFF
    table = _CRC_TABLE
    for b in data:
        crc = table[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _masked_crc(data: bytes) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _field(num: int, wt: int) -> bytes:
    return _varint((num << 3) | wt)


def encode_scalar_event(
    tag: str, step: int, wall_time: float, value: float, tensor: bool = False
) -> bytes:
    """Encode one Event proto holding a single scalar summary."""
    tag_b = tag.encode("utf-8")
    if tensor:
        tensor_b = (
            _field(1, 0) + _varint(_DT_FLOAT) + _field(4, 2) + _varint(4) + _F32.pack(value)
        )
        val = (
            _field(1, 2) + _varint(len(tag_b)) + tag_b
            + _field(8, 2) + _varint(len(tensor_b)) + tensor_b
        )
    else:
        val = _field(1, 2) + _varint(len(tag_b)) + tag_b + _field(2, 5) + _F32.pack(value)
    summary = _field(1, 2) + _varint(len(val)) + val
    return (
        _field(1, 1) + _F64.pack(wall_time)
        + _field(2, 0) + _varint(step & ((1 << 64) - 1))
        + _field(5, 2) + _varint(len(summary)) + summary
    )


def frame_record(payload: bytes) -> bytes:
    """Wrap an encoded Event in TFRecord framing (with valid masked CRCs)."""
    header = _U64.pack(len(payload))
    return (
        header
        + _U32.pack(_masked_crc(header))
        + payload
        + _U32.pack(_masked_crc(payload))
    )
