//! [`LogicalStatement`] -> [`PhysicalPlan`].
//!
//! For write-side and management statements the planning is one-to-one
//! (one logical statement, one operator). The interesting work lands
//! when SELECT joins the dispatch and the cost-model picks between
//! scan / index / post-filter / soft-filtered-ANN.

use crate::error::KovaQueryError;
use crate::logical::{LogicalStatement, LogicalVacuum};
use crate::physical::PhysicalPlan;

/// Pick the physical plan for a [`LogicalStatement`].
///
/// # Errors
///
/// Returns [`KovaQueryError::Plan`] for any statement the planner
/// doesn't yet know how to handle. As each statement's executor
/// support lands, its arm gets a real plan ; until then it errors
/// cleanly instead of panicking.
//
// By-value : real arms move fields out of LogicalStatement payloads
// when they land (same shape as the binder dispatch).
#[allow(clippy::needless_pass_by_value)]
pub fn plan(stmt: LogicalStatement) -> Result<PhysicalPlan, KovaQueryError> {
    match stmt {
        LogicalStatement::Checkpoint => Ok(PhysicalPlan::Checkpoint),
        LogicalStatement::Vacuum(LogicalVacuum { table }) => Ok(PhysicalPlan::Vacuum { table }),

        // Filled in as each statement gains executor support. Explicit
        // arms (rather than `_`) so the compiler errors the moment a
        // new LogicalStatement variant is added without a planner arm.
        LogicalStatement::Insert(_) => unimplemented("INSERT"),
        LogicalStatement::Update(_) => unimplemented("UPDATE"),
        LogicalStatement::Delete(_) => unimplemented("DELETE"),
        LogicalStatement::Query(_) => unimplemented("SELECT"),
    }
}

fn unimplemented(name: &str) -> Result<PhysicalPlan, KovaQueryError> {
    Err(KovaQueryError::Plan(format!(
        "planner not yet implemented for {name}"
    )))
}
