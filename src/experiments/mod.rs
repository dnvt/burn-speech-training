//! Variants from the experiment log that did not beat the baseline.
//!
//! Compiled only with `--features experiments`. They are kept so most runs in
//! `docs/experiment-log.md` can be re-run and extended, not because they are
//! recommended. (Run 5b, freezing the backbone for only the first epochs, is
//! not supported: `freeze_backbone` applies to the whole run.)
//!
//! Two of them had bugs in the original runs, fixed here:
//! distillation never trained the backbone, and rank regularization had
//! almost no gradient. Results from those runs are inconclusive.

pub mod finetune;
pub mod losses;
pub mod teacher;

pub use finetune::{
    finetune, ExperimentConfig, ScoreLoss, TeacherMode, WeightedTerm, WordWeighting,
};
