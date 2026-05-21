//! vector is the owned, dim-validated container of f32 components
//! it's the type every distance function, every index, every storage layer will take by reference.

use crate::KovaError;

/// The `Vector` type is the owned, dim-validated container of `f32` components.
///  It's the type every distance function, every index, every storage layer will take by reference.
#[derive(Debug, Clone, PartialEq)]
//box instead of vec to ensure heap allocation and fixed size after creation
// also Box<[T]> is more memory efficient than Vec<T> for fixed-size arrays, as it doesn't store capacity or allow resizing
pub struct Vector(Box<[f32]>);

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
}
