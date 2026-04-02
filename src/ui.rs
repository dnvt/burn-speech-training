//! Simple logging helpers replacing the original CLI UI functions.
//! These use tracing so consumers can control output via subscriber config.
pub fn section(msg: &str) {
    tracing::info!("=== {} ===", msg);
}

pub fn step(msg: &str) {
    tracing::info!("  {}", msg);
}

pub fn detail(msg: &str) {
    tracing::info!("    {}", msg);
}

pub fn success(msg: &str) {
    tracing::info!("  [ok] {}", msg);
}
