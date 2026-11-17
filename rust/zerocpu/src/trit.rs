//! Q-delta ternary quantization and the A6P-15A bitplane layout.
//!
//! The Python module `ultragraph.zerocpu.trit` is the reference implementation;
//! `tests/trit_parity.rs` pins this file to the golden vectors it produces.
//!
//! No floating point: `delta` is a Q16.16 threshold and every step is exact.

use std::fmt;

/// The three ternary values.
pub const TRIT_NEG: i8 = -1;
pub const TRIT_ZERO: i8 = 0;
pub const TRIT_ONE: i8 = 1;

/// A6P detail levels: level 0 is coarsest, level 14 the finest.
pub const A6P_LEVELS: usize = 15;
/// Fifteen magnitude planes plus one sign plane.
pub const A6P_PLANES: usize = A6P_LEVELS + 1;
/// Largest magnitude representable in the magnitude planes.
pub const MAGNITUDE_MAX: u64 = (1 << A6P_LEVELS) - 1;
/// At most sixteen lanes per vector: every plane is one `u16`.
pub const MAX_LANES: usize = 16;

/// Everything that can go wrong in this module.
#[derive(Debug, PartialEq, Eq)]
pub enum TritError {
    /// The A6P level was outside `0..A6P_LEVELS`.
    Level(i32),
    /// The threshold was negative.
    Delta(i64),
    /// More than [`MAX_LANES`] values were given.
    Width(usize),
    /// The plane list was not [`A6P_PLANES`] long.
    Planes(usize),
}

impl fmt::Display for TritError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TritError::Level(level) => {
                write!(f, "level must be in 0..{}, got {}", A6P_LEVELS - 1, level)
            }
            TritError::Delta(delta) => write!(f, "delta must be >= 0, got {delta}"),
            TritError::Width(width) => write!(f, "at most {MAX_LANES} lanes, got {width}"),
            TritError::Planes(count) => {
                write!(f, "expected {A6P_PLANES} planes, got {count}")
            }
        }
    }
}

impl std::error::Error for TritError {}

/// Threshold for A6P detail `level`: `16384` at 0 down to `1` at 14.
pub fn delta_for_level(level: usize) -> Result<i64, TritError> {
    if level >= A6P_LEVELS {
        return Err(TritError::Level(level as i32));
    }
    Ok(1i64 << (A6P_LEVELS - 1 - level))
}

/// Ternary threshold: `+1` at or above `delta`, `-1` at or below `-delta`.
///
/// `delta == 0` degenerates to the plain sign, with `0` still mapping to `0`.
pub fn quantize(value: i64, delta: i64) -> Result<i8, TritError> {
    if delta < 0 {
        return Err(TritError::Delta(delta));
    }
    if delta == 0 {
        return Ok(if value == 0 {
            TRIT_ZERO
        } else if value > 0 {
            TRIT_ONE
        } else {
            TRIT_NEG
        });
    }
    Ok(if value >= delta {
        TRIT_ONE
    } else if value <= -delta {
        TRIT_NEG
    } else {
        TRIT_ZERO
    })
}

/// [`quantize`] over a vector.
pub fn quantize_many(values: &[i64], delta: i64) -> Result<Vec<i8>, TritError> {
    let mut out = Vec::with_capacity(values.len());
    for &value in values {
        out.push(quantize(value, delta)?);
    }
    Ok(out)
}

/// Shrink toward zero by `1 - 2**-shift`.
///
/// The integer stand-in for `v *= exp(-dt)`: `shift == 1` halves, `shift == 2`
/// quarters.  A shift of 63 or more decays everything to zero.
pub fn decay(value: i64, shift: u32) -> i64 {
    if shift >= 63 || value == 0 {
        return 0;
    }
    let lost = (value.unsigned_abs() >> shift) as i64;
    if value > 0 {
        value - lost
    } else {
        value + lost
    }
}

/// Explode up to [`MAX_LANES`] values into [`A6P_PLANES`] words.
///
/// Plane `l` (0..14) carries bit `l` of each magnitude; plane 15 carries the
/// sign.  Magnitudes saturate at [`MAGNITUDE_MAX`] and missing lanes are zero.
pub fn bitplanes(values: &[i64]) -> Result<Vec<u16>, TritError> {
    if values.len() > MAX_LANES {
        return Err(TritError::Width(values.len()));
    }
    let mut planes = vec![0u16; A6P_PLANES];
    for (lane, &value) in values.iter().enumerate() {
        let bit = 1u16 << lane;
        let mut magnitude = value.unsigned_abs().min(MAGNITUDE_MAX);
        let mut level = 0;
        while magnitude != 0 {
            if magnitude & 1 == 1 {
                planes[level] |= bit;
            }
            magnitude >>= 1;
            level += 1;
        }
        if value < 0 {
            planes[A6P_LEVELS] |= bit;
        }
    }
    Ok(planes)
}

/// Rebuild sixteen signed values from the A6P-15A words.
pub fn from_planes(planes: &[u16]) -> Result<Vec<i64>, TritError> {
    if planes.len() != A6P_PLANES {
        return Err(TritError::Planes(planes.len()));
    }
    let sign_plane = planes[A6P_LEVELS];
    let mut out = Vec::with_capacity(MAX_LANES);
    for lane in 0..MAX_LANES {
        let bit = 1u16 << lane;
        let mut magnitude = 0i64;
        for level in 0..A6P_LEVELS {
            if planes[level] & bit != 0 {
                magnitude |= 1i64 << level;
            }
        }
        out.push(if sign_plane & bit != 0 {
            -magnitude
        } else {
            magnitude
        });
    }
    Ok(out)
}

/// XOR fold of every plane: a 16-bit checksum over the whole vector.
pub fn plane_parity(planes: &[u16]) -> u16 {
    planes.iter().fold(0u16, |acc, &word| acc ^ word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_ladder_descends_to_one() {
        assert_eq!(delta_for_level(0).unwrap(), 16384);
        assert_eq!(delta_for_level(A6P_LEVELS - 1).unwrap(), 1);
        assert_eq!(delta_for_level(A6P_LEVELS), Err(TritError::Level(15)));
    }

    #[test]
    fn quantize_boundaries_match_python() {
        assert_eq!(quantize(16384, 16384).unwrap(), TRIT_ONE);
        assert_eq!(quantize(16383, 16384).unwrap(), TRIT_ZERO);
        assert_eq!(quantize(-16384, 16384).unwrap(), TRIT_NEG);
        assert_eq!(quantize(-16383, 16384).unwrap(), TRIT_ZERO);
        assert_eq!(quantize(0, 0).unwrap(), TRIT_ZERO);
        assert_eq!(quantize(1, 0).unwrap(), TRIT_ONE);
        assert_eq!(quantize(-1, -1), Err(TritError::Delta(-1)));
    }

    #[test]
    fn decay_halves_toward_zero() {
        assert_eq!(decay(1000, 1), 500);
        assert_eq!(decay(1000, 2), 750);
        assert_eq!(decay(-1000, 1), -500);
        assert_eq!(decay(-1000, 2), -750);
        assert_eq!(decay(7, 1), 4);
        assert_eq!(decay(12345, 63), 0);
    }

    #[test]
    fn bitplanes_round_trip_and_saturate() {
        let values = [
            -32767i64, -1, 0, 1, 32767, -12345, 5432, 0, 1, -1, 2, -2, 3, -3, 4, -4,
        ];
        let planes = bitplanes(&values).unwrap();
        assert_eq!(planes.len(), A6P_PLANES);
        assert_eq!(from_planes(&planes).unwrap(), values);
        let huge = bitplanes(&[10_000_000_000, -10_000_000_000]).unwrap();
        let back = from_planes(&huge).unwrap();
        assert_eq!(back[0], 32767);
        assert_eq!(back[1], -32767);
    }

    #[test]
    fn width_and_plane_errors() {
        assert_eq!(bitplanes(&[0; 17]), Err(TritError::Width(17)));
        assert_eq!(from_planes(&[0; 15]), Err(TritError::Planes(15)));
        assert_eq!(plane_parity(&[0u16; 16]), 0);
    }
}
