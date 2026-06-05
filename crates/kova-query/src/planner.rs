//! [`LogicalStatement`] -> [`PhysicalPlan`].
//!
//! For write-side and management statements the planning is one-to-one
//! (one logical statement, one operator). The interesting work lands
//! when SELECT joins the dispatch and the cost-model picks between
//! scan / index / post-filter / soft-filtered-ANN.

use kova_core::VectorId;

use crate::error::KovaQueryError;
use crate::logical::{
    LogicalDelete, LogicalInsert, LogicalInsertSource, LogicalStatement, LogicalVacuum,
};
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
        LogicalStatement::Insert(LogicalInsert { table, rows }) => match rows {
            LogicalInsertSource::Single {
                id,
                embedding,
                metadata,
            } => Ok(PhysicalPlan::InsertOne {
                table,
                id,
                embedding,
                metadata,
            }),
            LogicalInsertSource::Batch { param } => Ok(PhysicalPlan::InsertMany {
                table,
                batch: param,
            }),
        },
        LogicalStatement::Delete(LogicalDelete {
            table,
            single_id_hint,
            predicate: _,
        }) => match single_id_hint {
            // Hint set : binder spotted `WHERE id = <integer-literal>`.
            // Skip straight to the fast path ; no predicate evaluation
            // needed.
            Some(id) => Ok(PhysicalPlan::DeleteById {
                table,
                id: VectorId::new(id),
            }),
            // Hint missing : predicate is param-bound, compound, or
            // doesn't match the simple-id shape. Full DELETE-by-predicate
            // is its own milestone (needs metadata scan + delete_many).
            None => Err(KovaQueryError::Plan(
                "DELETE WHERE <predicate> is not yet supported ; v1 supports \
                 DELETE WHERE id = <integer-literal> only"
                    .into(),
            )),
        },

        // Filled in as each statement gains executor support. Explicit
        // arms (rather than `_`) so the compiler errors the moment a
        // new LogicalStatement variant is added without a planner arm.
        LogicalStatement::Update(_) => unimplemented("UPDATE"),
        LogicalStatement::Query(_) => unimplemented("SELECT"),
    }
}

fn unimplemented(name: &str) -> Result<PhysicalPlan, KovaQueryError> {
    Err(KovaQueryError::Plan(format!(
        "planner not yet implemented for {name}"
    )))
}
