//! team048_accel: Rust kernels for the team 048 solution. Pure-Rust core (`kin`, `col`, `perc`) plus optional
//! PyO3 bindings (`--features python`, built by maturin). The Python package keeps a NumPy reference
//! implementation of everything here and falls back to it automatically if this module is missing.
pub mod col;
pub mod kin;
pub mod perc;
#[cfg(feature = "python")]
mod py;
