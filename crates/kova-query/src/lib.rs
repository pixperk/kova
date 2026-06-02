//! KQL : the query language for Kova.
//!
//! Hybrid vector + metadata search expressed as SQL-shaped statements
//! over a single `vectors` table. Covers SELECT, INSERT, UPDATE,
//! DELETE, VACUUM, CHECKPOINT, and (in v2) CREATE / DROP INDEX.
//!
//! # Pipeline
//!
//! ```text
//!   String
//!     │  parse_str
//!     ▼
//!   AstStatement       // permissive : syntax only
//!     │  bind
//!     ▼
//!   LogicalStatement   // typed, normalised
//!     │  plan
//!     ▼
//!   PhysicalPlan       // operator tree
//!     │  execute
//!     ▼
//!   Rows
//! ```
//!
//! Each arrow is its own module so each stage is independently
//! testable and the failure modes are localised.

pub mod ast;
pub mod error;
pub mod parser;

pub use ast::AstStatement;
pub use error::KovaQueryError;
pub use parser::parse_str;
