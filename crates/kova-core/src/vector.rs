//! vector is the owned, dim-validated container of f32 components
//! it's the type every distance function, every index, every storage layer will take by reference.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::KovaError;

/// The `Vector` type is the owned, dim-validated container of `f32` components.
///  It's the type every distance function, every index, every storage layer will take by reference.
#[derive(Debug, Clone, PartialEq)]
//box instead of vec to ensure heap allocation and fixed size after creation
// also Box<[T]> is more memory efficient than Vec<T> for fixed-size arrays, as it doesn't store capacity or allow resizing
pub struct Vector(Box<[f32]>);

// Hand-rolled serde so deserialisation routes through `try_new`, preserving
// the "non-empty + finite" invariant even for bytes coming off disk.
impl Serialize for Vector {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        self.as_slice().serialize(ser)
    }
}

impl<'de> Deserialize<'de> for Vector {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let data = Vec::<f32>::deserialize(de)?;
        Self::try_new(data).map_err(serde::de::Error::custom)
    }
}

impl Vector {
    /// Creates a new `Vector` from the given data, validating that it is non-empty and contains only finite values.
    /// `Into<Box<[f32]>>` allows us to accept `Vec<f32>` and convert it to `Box<[f32]>` without extra allocation, and also accept `Box<[f32]>` directly.
    /// Returns a `KovaError` if the input data is empty or contains any non-finite values (NaN or infinity).
    pub fn try_new(data: impl Into<Box<[f32]>>) -> Result<Self, KovaError> {
        let data = data.into();
        if data.is_empty() {
            return Err(KovaError::EmptyVector);
        }
        for (i, &component) in data.iter().enumerate() {
            if !component.is_finite() {
                return Err(KovaError::NonFinite {
                    index: i,
                    value: component,
                });
            }
        }
        Ok(Vector(data))
    }

    /// `dim` returns the number of components in the vector, which is the length of the inner `Box<[f32]>` slice.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// `as_slice` returns a reference to the inner slice of `f32` components, allowing read-only access to the vector's data.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation() {
        let vec = Vector::try_new(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(vec.dim(), 3);
        assert_eq!(vec.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vector_empty() {
        let err = Vector::try_new(Vec::new()).unwrap_err();
        assert!(matches!(err, KovaError::EmptyVector));
    }

    #[test]
    fn test_vector_non_finite() {
        let err = Vector::try_new(vec![1.0, f32::NAN, 3.0]).unwrap_err();
        assert!(matches!(err, KovaError::NonFinite { index: 1, value } if value.is_nan()));
    }

    #[test]
    fn test_vector_non_finite_infinity() {
        let err = Vector::try_new(vec![1.0, f32::INFINITY, 3.0]).unwrap_err();
        assert!(matches!(err, KovaError::NonFinite { index: 1, value } if value.is_infinite()));
    }

    #[test]
    fn vector_bincode_roundtrip() {
        let v = Vector::try_new(vec![1.0, 2.0, 3.0]).unwrap();
        let bytes = bincode::serialize(&v).expect("serialize");
        let decoded: Vector = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(decoded, v);
    }

    #[test]
    fn vector_deserialize_rejects_nan() {
        // Encode as a bare `Vec<f32>` containing NaN, then try to decode as
        // `Vector`. The hand-rolled Deserialize routes through `try_new`,
        // which must reject.
        let bad: Vec<f32> = vec![1.0, f32::NAN, 3.0];
        let bytes = bincode::serialize(&bad).unwrap();
        let result: Result<Vector, _> = bincode::deserialize(&bytes);
        assert!(
            result.is_err(),
            "expected NaN to be rejected on deserialize"
        );
    }

    #[test]
    fn vector_deserialize_rejects_empty() {
        let bad: Vec<f32> = Vec::new();
        let bytes = bincode::serialize(&bad).unwrap();
        let result: Result<Vector, _> = bincode::deserialize(&bytes);
        assert!(
            result.is_err(),
            "expected empty to be rejected on deserialize"
        );
    }
}
