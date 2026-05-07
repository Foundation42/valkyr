//! Tiny utilities shared across the project.

/// `(num + den - 1) / den`. The boring ceiling-divide for u32 dispatch
/// counts. Lifted out of three+ duplicates so the "where do we count
/// workgroups?" search has one obvious target.
pub fn ceilDiv(num: u32, den: u32) u32 {
    return (num + den - 1) / den;
}
