//! KQL : the query language for Kova.
//!
//! Hybrid vector + metadata search expressed as SQL-shaped statements
//! over a single `vectors` table. Covers SELECT, INSERT, UPDATE,
//! DELETE, VACUUM, CHECKPOINT, CREATE / DROP INDEX.

pub mod ast;
pub mod error;
pub mod parser;

pub use ast::AstStatement;
pub use error::KovaQueryError;
pub use parser::parse_str;
