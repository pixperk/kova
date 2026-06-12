//! Predicate-tree evaluation through the secondary-index catalog.
//!
//! The executor's `MetadataScan` arm calls [`try_index_eval`] to ask
//! "can the catalog answer this WHERE clause without walking every
//! metadata bag?" Three answers come back :
//!
//! - [`IndexEval::Full`] : the whole predicate was satisfied through
//!   indexes ; the bitmap IS the candidate set. The caller just
//!   iterates the ids and emits hits.
//! - [`IndexEval::Hybrid`] : indexes shrank the candidate set to a
//!   `bitmap`, but a `residue` predicate still has to be evaluated
//!   per-row on those candidates. Faster than a full scan, slower
//!   than a Full hit.
//! - [`IndexEval::Fallback`] : the catalog can't help. The caller
//!   runs the original `shard.scan_metadata(|m| eval_predicate(...))`
//!   path.
//!
//! The walk is structural :
//!
//! - **Atom** : translate `PredAtom` → `(field, IndexAtom)` and ask
//!   the catalog. Hits become `Full` ; misses (no index, unsupported
//!   atom, subscripted field, NULL on the RHS, distance threshold)
//!   become `Fallback`.
//! - **AND** : intersect the bitmaps of indexable children, collect
//!   the rest as a residue. At least one child must be indexable to
//!   leave Fallback.
//! - **OR** : every child must be `Full`. A single `Hybrid` or
//!   `Fallback` poisons the whole OR (we'd produce a candidate set
//!   that's missing rows the residue would have caught).
//! - **NOT, True, False** : `Fallback`. NOT could be supported as
//!   `all_live_ids - inner_bitmap` once we have an "all live ids"
//!   surface ; not in this slice.

use kova_core::Value;
use kova_meta_index::{CmpOp as IdxCmpOp, IndexAtom, IndexCatalog};
use roaring::RoaringTreemap;

use crate::ast::CmpOp as AstCmpOp;
use crate::executor::{ParamBindings, ParamValue};
use crate::logical::{BoundExpr, BoundLiteral, PredAtom, PredicateExpr};

/// Result of asking the catalog whether it can answer a predicate.
///
/// See the module docs for the semantics of each variant.
#[derive(Debug)]
pub(crate) enum IndexEval {
    /// Every atom in the predicate was answered by an index. The
    /// caller iterates the bitmap and emits hits without evaluating
    /// the predicate per-row.
    Full(RoaringTreemap),

    /// Indexes shrank the candidate set ; the residue still needs
    /// per-row evaluation against each candidate's metadata bag.
    Hybrid {
        /// Candidate ids produced by the indexable atoms.
        candidates: RoaringTreemap,
        /// Predicate slice that wasn't indexable. Run per-row against
        /// each candidate's bag.
        residue: PredicateExpr,
    },

    /// Catalog can't help. Caller runs the original scan path.
    Fallback,
}

/// Walk `pred` against `catalog`. See [`IndexEval`] for the result
/// shapes.
pub(crate) fn try_index_eval(
    pred: &PredicateExpr,
    catalog: &IndexCatalog,
    params: &ParamBindings,
) -> IndexEval {
    match pred {
        PredicateExpr::Atom(a) => atom_eval(a, catalog, params),
        PredicateExpr::And(children) => and_eval(children, catalog, params),
        PredicateExpr::Or(children) => or_eval(children, catalog, params),
        // NOT and the constant nodes punt to the scan path.
        // NOT could be `all_live_ids - inner_bitmap` once we expose
        // an "all live ids" bitmap ; True/False normally don't reach
        // MetadataScan (the binder folds them), but we cover them
        // defensively.
        PredicateExpr::Not(_) | PredicateExpr::True | PredicateExpr::False => IndexEval::Fallback,
    }
}

fn atom_eval(a: &PredAtom, catalog: &IndexCatalog, params: &ParamBindings) -> IndexEval {
    let Some((field, idx_atom)) = translate_atom(a, params) else {
        return IndexEval::Fallback;
    };
    match catalog.lookup(field, &idx_atom) {
        Some(bitmap) => IndexEval::Full(bitmap),
        None => IndexEval::Fallback,
    }
}

/// Translate a `PredAtom` into a `(field, IndexAtom)` pair the
/// catalog can use. Returns `None` for any atom shape the catalog
/// can never answer :
///
/// - subscripted fields (`attrs['key']`) : catalog only indexes
///   top-level field names
/// - `BoundLiteral::Null` on the RHS : there's no `IndexAtom` for
///   "field equals NULL" ; `IS NULL` is handled by the binder as
///   `NOT IsNotNull` and never reaches us as an atom
/// - parameter slots with non-scalar values (Id, Vector, Metadata,
///   Batch) : wrong type for a predicate value
/// - `DistanceThreshold` : radius operator owns it, not the catalog
/// - `CmpOp::Eq` on the AST side never reaches here : the binder
///   routes equality to `PredAtom::Eq`, not `PredAtom::Cmp`
fn translate_atom<'a>(a: &'a PredAtom, params: &ParamBindings) -> Option<(&'a str, IndexAtom)> {
    match a {
        PredAtom::Eq { field, value } => {
            if field.subscript.is_some() {
                return None;
            }
            let v = resolve_to_value(value, params)?;
            Some((field.name.as_str(), IndexAtom::Eq(v)))
        }
        PredAtom::Cmp { field, op, value } => {
            if field.subscript.is_some() {
                return None;
            }
            let v = resolve_to_value(value, params)?;
            let idx_op = translate_cmp_op(*op)?;
            Some((field.name.as_str(), IndexAtom::Cmp(idx_op, v)))
        }
        PredAtom::In { field, values } => {
            if field.subscript.is_some() {
                return None;
            }
            let mut vs: Vec<Value> = Vec::with_capacity(values.len());
            for lit in values {
                vs.push(literal_to_value(lit)?);
            }
            Some((field.name.as_str(), IndexAtom::In(vs)))
        }
        PredAtom::Between { field, lo, hi } => {
            if field.subscript.is_some() {
                return None;
            }
            let lo_v = literal_to_value(lo)?;
            let hi_v = literal_to_value(hi)?;
            Some((field.name.as_str(), IndexAtom::Between(lo_v, hi_v)))
        }
        PredAtom::IsNotNull { field } => {
            if field.subscript.is_some() {
                return None;
            }
            Some((field.name.as_str(), IndexAtom::IsNotNull))
        }
        PredAtom::ArrayContains { field, value } => {
            if field.subscript.is_some() {
                return None;
            }
            let v = literal_to_value(value)?;
            Some((field.name.as_str(), IndexAtom::ArrayContains(v)))
        }
        PredAtom::DistanceThreshold { .. } => None,
    }
}

fn translate_cmp_op(op: AstCmpOp) -> Option<IdxCmpOp> {
    match op {
        // Eq is its own PredAtom variant ; if it shows up under Cmp
        // something upstream is wrong, but we silently punt rather
        // than panic.
        AstCmpOp::Eq => None,
        AstCmpOp::Lt => Some(IdxCmpOp::Lt),
        AstCmpOp::Le => Some(IdxCmpOp::Le),
        AstCmpOp::Gt => Some(IdxCmpOp::Gt),
        AstCmpOp::Ge => Some(IdxCmpOp::Ge),
        AstCmpOp::Ne => Some(IdxCmpOp::Ne),
    }
}

fn resolve_to_value(expr: &BoundExpr, params: &ParamBindings) -> Option<Value> {
    match expr {
        BoundExpr::Literal(l) => literal_to_value(l),
        BoundExpr::Param(p) => params.resolve(p).ok().and_then(param_to_value),
    }
}

fn literal_to_value(l: &BoundLiteral) -> Option<Value> {
    match l {
        BoundLiteral::String(s) => Some(Value::String(s.clone())),
        BoundLiteral::I64(n) => Some(Value::I64(*n)),
        BoundLiteral::F64(f) => Some(Value::F64(*f)),
        BoundLiteral::Bool(b) => Some(Value::Bool(*b)),
        // NULL has no Value representation through the index. The
        // catalog can't store "the absence of a key" as a key.
        BoundLiteral::Null => None,
    }
}

fn param_to_value(p: &ParamValue) -> Option<Value> {
    match p {
        ParamValue::String(s) => Some(Value::String(s.clone())),
        ParamValue::I64(n) => Some(Value::I64(*n)),
        ParamValue::F64(f) => Some(Value::F64(*f)),
        ParamValue::Bool(b) => Some(Value::Bool(*b)),
        // Id/Vector/Metadata/Batch are wrong type for a predicate
        // RHS ; Null has no value shape (see literal_to_value).
        ParamValue::Id(_)
        | ParamValue::Vector(_)
        | ParamValue::Metadata(_)
        | ParamValue::Batch(_)
        | ParamValue::Null => None,
    }
}

fn and_eval(
    children: &[PredicateExpr],
    catalog: &IndexCatalog,
    params: &ParamBindings,
) -> IndexEval {
    let mut candidates: Option<RoaringTreemap> = None;
    let mut residue: Vec<PredicateExpr> = Vec::new();

    for child in children {
        match try_index_eval(child, catalog, params) {
            IndexEval::Full(bm) => {
                candidates = Some(intersect(candidates, bm));
            }
            IndexEval::Hybrid {
                candidates: bm,
                residue: r,
            } => {
                candidates = Some(intersect(candidates, bm));
                residue.push(r);
            }
            IndexEval::Fallback => {
                residue.push(child.clone());
            }
        }
    }

    let Some(bm) = candidates else {
        // Nothing in the AND was indexable.
        return IndexEval::Fallback;
    };

    match residue.len() {
        0 => IndexEval::Full(bm),
        1 => IndexEval::Hybrid {
            candidates: bm,
            residue: residue.into_iter().next().expect("len == 1"),
        },
        _ => IndexEval::Hybrid {
            candidates: bm,
            residue: PredicateExpr::And(residue),
        },
    }
}

fn or_eval(
    children: &[PredicateExpr],
    catalog: &IndexCatalog,
    params: &ParamBindings,
) -> IndexEval {
    // OR is a unanimous-vote situation : every branch must produce
    // a Full bitmap. A single Hybrid or Fallback branch means we'd
    // be unable to enumerate rows that satisfy ONLY that branch,
    // and the OR result would be incomplete.
    let mut acc: Option<RoaringTreemap> = None;
    for child in children {
        match try_index_eval(child, catalog, params) {
            IndexEval::Full(bm) => {
                acc = Some(match acc {
                    Some(prev) => prev | bm,
                    None => bm,
                });
            }
            IndexEval::Hybrid { .. } | IndexEval::Fallback => return IndexEval::Fallback,
        }
    }
    match acc {
        Some(bm) => IndexEval::Full(bm),
        None => IndexEval::Fallback,
    }
}

fn intersect(prev: Option<RoaringTreemap>, next: RoaringTreemap) -> RoaringTreemap {
    match prev {
        Some(p) => p & next,
        None => next,
    }
}

#[cfg(test)]
mod tests {
    use kova_core::Value;
    use kova_meta_index::IndexCatalog;

    use super::*;
    use crate::executor::ParamBindings;
    use crate::logical::{BoundExpr, BoundLiteral, FieldRef, PredAtom, PredicateExpr};

    fn s(x: &str) -> Value {
        Value::String(x.into())
    }
    fn i(n: i64) -> Value {
        Value::I64(n)
    }
    fn lit_s(x: &str) -> BoundExpr {
        BoundExpr::Literal(BoundLiteral::String(x.into()))
    }
    fn lit_i(n: i64) -> BoundExpr {
        BoundExpr::Literal(BoundLiteral::I64(n))
    }
    fn field(name: &str) -> FieldRef {
        FieldRef::plain(name)
    }
    fn eq_atom(name: &str, value: BoundExpr) -> PredicateExpr {
        PredicateExpr::Atom(PredAtom::Eq {
            field: field(name),
            value,
        })
    }

    fn fresh_catalog(field: &str) -> IndexCatalog {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index(field);
        cat
    }

    fn id(n: u64) -> kova_core::VectorId {
        kova_core::VectorId::new(n)
    }

    fn meta_with(pairs: &[(&str, Value)]) -> kova_core::Metadata {
        let mut m = kova_core::Metadata::new();
        for (k, v) in pairs {
            m.insert((*k).to_string(), v.clone());
        }
        m
    }

    fn populate(cat: &mut IndexCatalog) {
        cat.on_insert(id(0), &meta_with(&[("category", s("docs"))]));
        cat.on_insert(id(1), &meta_with(&[("category", s("blog"))]));
        cat.on_insert(id(2), &meta_with(&[("category", s("docs"))]));
    }

    #[test]
    fn eq_on_indexed_field_returns_full() {
        let mut cat = fresh_catalog("category");
        populate(&mut cat);
        let pred = eq_atom("category", lit_s("docs"));
        let params = ParamBindings::empty();
        let r = try_index_eval(&pred, &cat, &params);
        match r {
            IndexEval::Full(bm) => {
                assert_eq!(bm.len(), 2);
                assert!(bm.contains(0));
                assert!(bm.contains(2));
            }
            other => panic!("expected Full, got {other:?}"),
        }
    }

    #[test]
    fn eq_on_unindexed_field_falls_back() {
        let cat = IndexCatalog::new(); // empty catalog
        let pred = eq_atom("category", lit_s("docs"));
        let params = ParamBindings::empty();
        assert!(matches!(
            try_index_eval(&pred, &cat, &params),
            IndexEval::Fallback
        ));
    }

    #[test]
    fn subscripted_field_falls_back() {
        let cat = fresh_catalog("attrs");
        let pred = PredicateExpr::Atom(PredAtom::Eq {
            field: FieldRef {
                name: "attrs".into(),
                subscript: Some("key".into()),
            },
            value: lit_s("docs"),
        });
        let params = ParamBindings::empty();
        assert!(matches!(
            try_index_eval(&pred, &cat, &params),
            IndexEval::Fallback
        ));
    }

    #[test]
    fn and_of_two_indexed_atoms_is_full() {
        let mut cat = IndexCatalog::new();
        cat.add_hash_index("category");
        cat.add_btree_index("year");
        cat.on_insert(
            id(0),
            &meta_with(&[("category", s("docs")), ("year", i(2024))]),
        );
        cat.on_insert(
            id(1),
            &meta_with(&[("category", s("blog")), ("year", i(2024))]),
        );
        cat.on_insert(
            id(2),
            &meta_with(&[("category", s("docs")), ("year", i(2020))]),
        );

        let pred = PredicateExpr::And(vec![
            eq_atom("category", lit_s("docs")),
            PredicateExpr::Atom(PredAtom::Cmp {
                field: field("year"),
                op: AstCmpOp::Ge,
                value: lit_i(2023),
            }),
        ]);
        let params = ParamBindings::empty();
        match try_index_eval(&pred, &cat, &params) {
            IndexEval::Full(bm) => {
                assert_eq!(bm.len(), 1);
                assert!(bm.contains(0));
            }
            other => panic!("expected Full, got {other:?}"),
        }
    }

    #[test]
    fn and_with_one_unindexed_atom_is_hybrid() {
        let mut cat = fresh_catalog("category");
        cat.on_insert(id(0), &meta_with(&[("category", s("docs"))]));
        cat.on_insert(id(1), &meta_with(&[("category", s("docs"))]));

        let pred = PredicateExpr::And(vec![
            eq_atom("category", lit_s("docs")),
            // `priority` isn't indexed.
            eq_atom("priority", lit_i(5)),
        ]);
        let params = ParamBindings::empty();
        match try_index_eval(&pred, &cat, &params) {
            IndexEval::Hybrid {
                candidates,
                residue,
            } => {
                assert_eq!(candidates.len(), 2);
                // Single residue lifts out of the And wrapper.
                match residue {
                    PredicateExpr::Atom(PredAtom::Eq { field: f, .. }) => {
                        assert_eq!(f.name, "priority");
                    }
                    other => panic!("expected residue atom on priority, got {other:?}"),
                }
            }
            other => panic!("expected Hybrid, got {other:?}"),
        }
    }

    #[test]
    fn and_with_all_unindexed_atoms_falls_back() {
        let cat = IndexCatalog::new();
        let pred = PredicateExpr::And(vec![eq_atom("foo", lit_s("a")), eq_atom("bar", lit_s("b"))]);
        let params = ParamBindings::empty();
        assert!(matches!(
            try_index_eval(&pred, &cat, &params),
            IndexEval::Fallback
        ));
    }

    #[test]
    fn or_of_all_indexed_atoms_is_full() {
        let mut cat = fresh_catalog("category");
        cat.on_insert(id(0), &meta_with(&[("category", s("docs"))]));
        cat.on_insert(id(1), &meta_with(&[("category", s("blog"))]));
        cat.on_insert(id(2), &meta_with(&[("category", s("code"))]));

        let pred = PredicateExpr::Or(vec![
            eq_atom("category", lit_s("docs")),
            eq_atom("category", lit_s("code")),
        ]);
        let params = ParamBindings::empty();
        match try_index_eval(&pred, &cat, &params) {
            IndexEval::Full(bm) => {
                assert_eq!(bm.len(), 2);
                assert!(bm.contains(0));
                assert!(bm.contains(2));
                assert!(!bm.contains(1));
            }
            other => panic!("expected Full, got {other:?}"),
        }
    }

    #[test]
    fn or_with_unindexed_branch_falls_back() {
        let mut cat = fresh_catalog("category");
        populate(&mut cat);

        let pred = PredicateExpr::Or(vec![
            eq_atom("category", lit_s("docs")),
            // priority isn't indexed
            eq_atom("priority", lit_i(5)),
        ]);
        let params = ParamBindings::empty();
        assert!(matches!(
            try_index_eval(&pred, &cat, &params),
            IndexEval::Fallback
        ));
    }

    #[test]
    fn not_falls_back_unconditionally() {
        let mut cat = fresh_catalog("category");
        populate(&mut cat);

        let pred = PredicateExpr::Not(Box::new(eq_atom("category", lit_s("docs"))));
        let params = ParamBindings::empty();
        assert!(matches!(
            try_index_eval(&pred, &cat, &params),
            IndexEval::Fallback
        ));
    }
}
