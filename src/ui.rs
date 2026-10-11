//! Progress logging helpers. Output goes through `tracing`, so nothing is
//! printed unless the caller installs a subscriber.
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
