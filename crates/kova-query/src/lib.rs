//! KQL : the query language for Kova.
//!
//! Hybrid vector + metadata search expressed as SQL-shaped statements
//! over a single `vectors` table. Covers SELECT, INSERT, UPDATE,
//! DELETE, VACUUM, CHECKPOINT, CREATE / DROP INDEX.

pub mod ast;
pub mod binder;
pub mod error;
pub mod executor;
pub mod logical;
pub mod parser;
pub mod physical;
pub mod planner;
pub mod printer;

pub use ast::AstStatement;
pub use binder::bind;
pub use error::KovaQueryError;
pub use executor::{Engine, ExecutionResult, ParamBindings, ParamValue};
pub use logical::LogicalStatement;
pub use parser::parse_str;
pub use physical::PhysicalPlan;
pub use planner::plan_with_estimator;
pub use printer::print;
