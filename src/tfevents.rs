//! Zero-copy reader (and writer, for tests/demos) of TensorBoard event files.
//!
//! Implements just enough of the TFRecord framing and the protobuf wire
//! format to extract scalar summaries at memory-bandwidth speed, with no
//! protobuf or TensorFlow dependency.  Both classic `simple_value` scalars
//! and TF2 tensor-encoded scalars are understood.

// TensorProto dtypes we can read a scalar out of.
const DT_FLOAT: u64 = 1;
const DT_DOUBLE: u64 = 2;
const DT_INT32: u64 = 3;
const DT_INT64: u64 = 9;

#[derive(Debug)]
pub struct ParseError;

type PResult<T> = Result<T, ParseError>;

#[inline]
fn read_varint(buf: &[u8], pos: &mut usize) -> PResult<u64> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    loop {
        let b = *buf.get(*pos).ok_or(ParseError)?;
        *pos += 1;
        result |= u64::from(b & 0x7F) << shift;
        if b < 0x80 {
            return Ok(result);
        }
        shift += 7;
        if shift > 63 {
            return Err(ParseError);
        }
    }
}

/// Fast path: field keys and short lengths are single-byte varints.
#[inline]
fn read_key(buf: &[u8], pos: &mut usize) -> PResult<u64> {
    let b = *buf.get(*pos).ok_or(ParseError)?;
    if b < 0x80 {
        *pos += 1;
        Ok(u64::from(b))
    } else {
        read_varint(buf, pos)
    }
}

#[inline]
fn skip_field(buf: &[u8], pos: &mut usize, wire_type: u64) -> PResult<()> {
    match wire_type {
        0 => {
            read_varint(buf, pos)?;
        }
        1 => *pos += 8,
        2 => {
            let ln = read_varint(buf, pos)? as usize;
            *pos += ln;
        }
        5 => *pos += 4,
        _ => return Err(ParseError),
    }
    if *pos > buf.len() {
        return Err(ParseError);
    }
    Ok(())
}

#[inline]
fn get_f32(buf: &[u8], pos: usize) -> PResult<f32> {
    let bytes: [u8; 4] = buf.get(pos..pos + 4).ok_or(ParseError)?.try_into().unwrap();
    Ok(f32::from_le_bytes(bytes))
}

#[inline]
fn get_f64(buf: &[u8], pos: usize) -> PResult<f64> {
    let bytes: [u8; 8] = buf.get(pos..pos + 8).ok_or(ParseError)?.try_into().unwrap();
    Ok(f64::from_le_bytes(bytes))
}

fn tensor_scalar(buf: &[u8]) -> PResult<Option<f64>> {
    let mut pos = 0usize;
    let mut dtype = 0u64;
    let mut content: Option<&[u8]> = None;
    let mut value: Option<f64> = None;
    while pos < buf.len() {
        let key = read_key(buf, &mut pos)?;
        match key {
            0x08 => dtype = read_varint(buf, &mut pos)?, // dtype: field 1, varint
            0x22 => {
                // tensor_content: field 4, bytes
                let ln = read_varint(buf, &mut pos)? as usize;
                content = Some(buf.get(pos..pos + ln).ok_or(ParseError)?);
                pos += ln;
            }
            0x2A => {
                // float_val: field 5, packed
                let ln = read_varint(buf, &mut pos)? as usize;
                if value.is_none() && ln >= 4 {
                    value = Some(f64::from(get_f32(buf, pos)?));
                }
                pos += ln;
            }
            0x2D => {
                // float_val: field 5, unpacked fixed32
                if value.is_none() {
                    value = Some(f64::from(get_f32(buf, pos)?));
                }
                pos += 4;
            }
            0x32 => {
                // double_val: field 6, packed
                let ln = read_varint(buf, &mut pos)? as usize;
                if value.is_none() && ln >= 8 {
                    value = Some(get_f64(buf, pos)?);
                }
                pos += ln;
            }
            0x31 => {
                // double_val: field 6, unpacked fixed64
                if value.is_none() {
                    value = Some(get_f64(buf, pos)?);
                }
                pos += 8;
            }
            0x3A | 0x52 => {
                // int_val (7) / int64_val (10), packed varints
                let ln = read_varint(buf, &mut pos)? as usize;
                let end = pos + ln;
                if value.is_none() && ln > 0 {
                    let mut p = pos;
                    let v = read_varint(buf, &mut p)?;
                    value = Some(v as i64 as f64);
                }
                pos = end;
            }
            0x38 | 0x50 => {
                let v = read_varint(buf, &mut pos)?;
                if value.is_none() {
                    value = Some(v as i64 as f64);
                }
            }
            _ => skip_field(buf, &mut pos, key & 7)?,
        }
        if pos > buf.len() {
            return Err(ParseError);
        }
    }
    if value.is_some() {
        return Ok(value);
    }
    if let Some(c) = content {
        let v = match dtype {
            DT_FLOAT if c.len() >= 4 => {
                Some(f64::from(f32::from_le_bytes(c[..4].try_into().unwrap())))
            }
            DT_DOUBLE if c.len() >= 8 => Some(f64::from_le_bytes(c[..8].try_into().unwrap())),
            DT_INT32 if c.len() >= 4 => Some(i32::from_le_bytes(c[..4].try_into().unwrap()) as f64),
            DT_INT64 if c.len() >= 8 => Some(i64::from_le_bytes(c[..8].try_into().unwrap()) as f64),
            _ => None,
        };
        return Ok(v);
    }
    Ok(None)
}

/// Parse one Summary.Value message; returns (tag bytes, scalar value) if both present.
fn parse_value(buf: &[u8]) -> PResult<Option<(&[u8], f64)>> {
    let mut pos = 0usize;
    let mut tag: Option<&[u8]> = None;
    let mut value: Option<f64> = None;
    while pos < buf.len() {
        let key = read_key(buf, &mut pos)?;
        match key {
            0x0A => {
                // tag: field 1, length-delimited
                let ln = read_varint(buf, &mut pos)? as usize;
                tag = Some(buf.get(pos..pos + ln).ok_or(ParseError)?);
                pos += ln;
            }
            0x15 => {
                // simple_value: field 2, fixed32
                value = Some(f64::from(get_f32(buf, pos)?));
                pos += 4;
            }
            0x42 => {
                // tensor: field 8, length-delimited
                let ln = read_varint(buf, &mut pos)? as usize;
                let sub = buf.get(pos..pos + ln).ok_or(ParseError)?;
                if value.is_none() {
                    value = tensor_scalar(sub)?;
                }
                pos += ln;
            }
            _ => skip_field(buf, &mut pos, key & 7)?,
        }
    }
    Ok(match (tag, value) {
        (Some(t), Some(v)) => Some((t, v)),
        _ => None,
    })
}

/// Parse one Event message, feeding scalar points into `sink(tag, step, wall_time, value)`.
fn parse_event(buf: &[u8], sink: &mut impl FnMut(&[u8], i64, f64, f64)) -> PResult<()> {
    let mut pos = 0usize;
    let mut wall_time = 0.0f64;
    let mut step = 0i64;
    let mut summary: Option<&[u8]> = None;
    while pos < buf.len() {
        let key = read_key(buf, &mut pos)?;
        match key {
            0x09 => {
                // wall_time: field 1, fixed64 double
                wall_time = get_f64(buf, pos)?;
                pos += 8;
            }
            0x10 => {
                // step: field 2, varint int64
                step = read_varint(buf, &mut pos)? as i64;
            }
            0x2A => {
                // summary: field 5, length-delimited
                let ln = read_varint(buf, &mut pos)? as usize;
                summary = Some(buf.get(pos..pos + ln).ok_or(ParseError)?);
                pos += ln;
            }
            _ => skip_field(buf, &mut pos, key & 7)?,
        }
    }
    let Some(sum) = summary else { return Ok(()) };
    let mut pos = 0usize;
    while pos < sum.len() {
        let key = read_key(sum, &mut pos)?;
        if key == 0x0A {
            // repeated Summary.value
            let ln = read_varint(sum, &mut pos)? as usize;
            let sub = sum.get(pos..pos + ln).ok_or(ParseError)?;
            if let Some((tag, value)) = parse_value(sub)? {
                sink(tag, step, wall_time, value);
            }
            pos += ln;
        } else {
            skip_field(sum, &mut pos, key & 7)?;
        }
    }
    Ok(())
}

/// Outcome of parsing a chunk of a tfevents file.
pub struct ChunkResult {
    /// Bytes of complete, valid records processed; resume from here next time.
    pub consumed: usize,
    /// A record failed to decode; the rest of the file should be abandoned.
    pub corrupt: bool,
}

/// Walk TFRecord frames in `data`, feeding scalars into `sink`.
///
/// A trailing partially-written record (a file that is still being appended
/// to) is left unconsumed so the caller can retry from the same offset once
/// more bytes exist.  CRCs are not verified — framing and protobuf structure
/// are validation enough for this use, and skipping them is much faster.
pub fn parse_chunk(data: &[u8], sink: &mut impl FnMut(&[u8], i64, f64, f64)) -> ChunkResult {
    let mut pos = 0usize;
    while pos + 12 <= data.len() {
        let length = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        if length > (1 << 31) {
            return ChunkResult { consumed: pos, corrupt: true };
        }
        let body = pos + 12;
        let after = body + length + 4;
        if after > data.len() {
            break; // truncated / still being written
        }
        if parse_event(&data[body..body + length], sink).is_err() {
            return ChunkResult { consumed: pos, corrupt: true };
        }
        pos = after;
    }
    ChunkResult { consumed: pos, corrupt: false }
}

// --------------------------------------------------------------------------
// Writer (tests and the demo-log generator)
// --------------------------------------------------------------------------

fn crc32c(data: &[u8]) -> u32 {
    // Small table-based CRC32-C; only used when *writing* demo logs.
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { (c >> 1) ^ 0x82F6_3B78 } else { c >> 1 };
            }
            *e = c;
        }
        t
    });
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = table[((crc ^ u32::from(b)) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

fn masked_crc(data: &[u8]) -> u32 {
    let crc = crc32c(data);
    (crc.rotate_right(15)).wrapping_add(0xA282_EAD8)
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            out.push(b | 0x80);
        } else {
            out.push(b);
            return;
        }
    }
}

fn put_field(out: &mut Vec<u8>, num: u64, wt: u64) {
    put_varint(out, (num << 3) | wt);
}

/// Encode one Event proto holding a single scalar summary.
pub fn encode_scalar_event(
    tag: &str,
    step: i64,
    wall_time: f64,
    value: f32,
    tensor: bool,
) -> Vec<u8> {
    let mut val = Vec::with_capacity(tag.len() + 24);
    put_field(&mut val, 1, 2);
    put_varint(&mut val, tag.len() as u64);
    val.extend_from_slice(tag.as_bytes());
    if tensor {
        let mut t = Vec::with_capacity(12);
        put_field(&mut t, 1, 0);
        put_varint(&mut t, DT_FLOAT);
        put_field(&mut t, 4, 2);
        put_varint(&mut t, 4);
        t.extend_from_slice(&value.to_le_bytes());
        put_field(&mut val, 8, 2);
        put_varint(&mut val, t.len() as u64);
        val.extend_from_slice(&t);
    } else {
        put_field(&mut val, 2, 5);
        val.extend_from_slice(&value.to_le_bytes());
    }

    let mut summary = Vec::with_capacity(val.len() + 4);
    put_field(&mut summary, 1, 2);
    put_varint(&mut summary, val.len() as u64);
    summary.extend_from_slice(&val);

    let mut ev = Vec::with_capacity(summary.len() + 24);
    put_field(&mut ev, 1, 1);
    ev.extend_from_slice(&wall_time.to_le_bytes());
    put_field(&mut ev, 2, 0);
    put_varint(&mut ev, step as u64);
    put_field(&mut ev, 5, 2);
    put_varint(&mut ev, summary.len() as u64);
    ev.extend_from_slice(&summary);
    ev
}

/// Wrap an encoded Event in TFRecord framing (with valid masked CRCs).
pub fn frame_record(payload: &[u8]) -> Vec<u8> {
    let header = (payload.len() as u64).to_le_bytes();
    let mut out = Vec::with_capacity(payload.len() + 16);
    out.extend_from_slice(&header);
    out.extend_from_slice(&masked_crc(&header).to_le_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&masked_crc(payload).to_le_bytes());
    out
}

// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn record(tag: &str, step: i64, wall: f64, value: f32, tensor: bool) -> Vec<u8> {
        frame_record(&encode_scalar_event(tag, step, wall, value, tensor))
    }

    fn collect(data: &[u8]) -> (Vec<(String, i64, f64, f64)>, ChunkResult) {
        let mut pts = Vec::new();
        let res = parse_chunk(data, &mut |tag, step, wall, val| {
            pts.push((String::from_utf8_lossy(tag).into_owned(), step, wall, val));
        });
        (pts, res)
    }

    #[test]
    fn simple_value_roundtrip() {
        let mut data = record("loss", 10, 123.5, 0.25, false);
        data.extend(record("acc", 11, 124.0, 0.75, false));
        let (pts, res) = collect(&data);
        assert_eq!(res.consumed, data.len());
        assert!(!res.corrupt);
        assert_eq!(pts.len(), 2);
        assert_eq!(pts[0].0, "loss");
        assert_eq!(pts[0].1, 10);
        assert!((pts[0].2 - 123.5).abs() < 1e-9);
        assert!((pts[0].3 - 0.25).abs() < 1e-6);
        assert_eq!(pts[1].0, "acc");
    }

    #[test]
    fn tensor_scalar_roundtrip() {
        let data = record("loss", 5, 1.0, 4.0625, true);
        let (pts, res) = collect(&data);
        assert_eq!(res.consumed, data.len());
        assert_eq!(pts.len(), 1);
        assert!((pts[0].3 - 4.0625).abs() < 1e-4);
    }

    #[test]
    fn truncated_tail_left_unconsumed() {
        let full = record("loss", 1, 1.0, 0.5, false);
        let partial = record("loss", 2, 2.0, 0.6, false);
        let mut data = full.clone();
        data.extend_from_slice(&partial[..partial.len() - 7]);
        let (pts, res) = collect(&data);
        assert_eq!(pts.len(), 1);
        assert_eq!(res.consumed, full.len());
        assert!(!res.corrupt);
        // once the rest arrives, parsing resumes from that offset
        let mut all = full.clone();
        all.extend(partial.clone());
        let (pts2, res2) = collect(&all[res.consumed..]);
        assert_eq!(pts2.len(), 1);
        assert_eq!(pts2[0].1, 2);
        assert_eq!(res2.consumed, partial.len());
    }

    #[test]
    fn event_without_summary_skipped() {
        // a file_version-style event: wall_time only
        let mut ev = Vec::new();
        put_field(&mut ev, 1, 1);
        ev.extend_from_slice(&42.0f64.to_le_bytes());
        let mut data = frame_record(&ev);
        data.extend(record("loss", 1, 1.0, 0.5, false));
        let (pts, res) = collect(&data);
        assert_eq!(pts.len(), 1);
        assert_eq!(res.consumed, data.len());
    }

    #[test]
    fn corrupt_record_reported() {
        let mut data = record("loss", 1, 1.0, 0.5, false);
        let good = data.len();
        let mut bad = record("loss", 2, 2.0, 0.6, false);
        // wire type 3 (start-group) inside the event payload is rejected
        bad[12] = 0x0B;
        data.extend(bad);
        let (pts, res) = collect(&data);
        assert_eq!(pts.len(), 1);
        assert!(res.corrupt);
        assert_eq!(res.consumed, good);
    }

    #[test]
    fn negative_and_large_values() {
        let mut data = record("neg", 1, 1.0, -123.456, false);
        data.extend(record("big", 2, 2.0, 1e30, false));
        let (pts, _) = collect(&data);
        assert!((pts[0].3 + 123.456).abs() < 1e-2);
        assert!((pts[1].3 / 1e30 - 1.0).abs() < 1e-3);
    }
}
