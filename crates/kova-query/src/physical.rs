//! Physical plan : the executor's IR.
//!
//! Operator tree the executor walks against a `Shard`. Where the
//! [`crate::logical::LogicalStatement`] captures *what to compute*,
//! the physical plan captures *how* : which operator runs, in what
//! order, with which parameter slots resolved at execute time.
//!
//! Operators land incrementally. CHECKPOINT is first ; INSERT /
//! DELETE / VACUUM / SELECT follow.

/// Physical operator. v1 grows this enum as each statement gets its
/// executor support. Explicit variants (no catchall) so the executor's
/// dispatch is exhaustive and the compiler complains when an arm goes
/// missing.
#[derive(Debug, Clone)]
pub enum PhysicalPlan {
    /// Stop-the-world checkpoint : vacuum + WAL fsync + snapshot
    /// write + manifest commit + WAL truncate. Dispatches directly
    /// to `Shard::checkpoint`. Returns the committed LSN.
    Checkpoint,
}
