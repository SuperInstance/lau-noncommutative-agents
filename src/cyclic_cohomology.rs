//! Cyclic cohomology.

use nalgebra::{DMatrix, Complex};
use crate::algebra::CStarAlgebra;

/// Cyclic cohomology computation.
pub struct CyclicCohomology {
    pub algebra_dim: usize,
}

impl CyclicCohomology {
    pub fn new(algebra: &CStarAlgebra) -> Self { Self { algebra_dim: algebra.dimension } }

    /// The trace cocycle (0-dimensional cyclic cohomology).
    pub fn trace_cocycle(a: &DMatrix<Complex<f64>>) -> Complex<f64> { a.trace() }

    /// Fundamental cyclic 2-cocycle: φ(a₀, a₁, a₂) = Tr(a₀[D, a₁][D, a₂]).
    pub fn fundamental_2_cocycle(elements: &[DMatrix<Complex<f64>>], dirac: &DMatrix<Complex<f64>>) -> Complex<f64> {
        if elements.len() < 3 { return Complex::new(0.0, 0.0); }
        let comm1 = dirac * &elements[1] - &elements[1] * dirac;
        let comm2 = dirac * &elements[2] - &elements[2] * dirac;
        (&elements[0] * &comm1 * &comm2).trace()
    }

    /// HC⁰(A): traces.
    pub fn hc0(elements: &[DMatrix<Complex<f64>>]) -> Vec<Complex<f64>> {
        elements.iter().map(|a| a.trace()).collect()
    }

    /// Check cyclic property.
    pub fn is_cyclic_cocycle<F>(phi: F, n: usize, elements: &[DMatrix<Complex<f64>>], tol: f64) -> bool
    where F: Fn(&[DMatrix<Complex<f64>>]) -> Complex<f64> {
        if elements.len() < n + 1 { return true; }
        let args: Vec<_> = (0..=n).map(|i| elements[i].clone()).collect();
        let val_forward = phi(&args);
        let mut rotated: Vec<_> = (1..=n).map(|i| elements[i].clone()).collect();
        rotated.push(elements[0].clone());
        let val_rotated = phi(&rotated);
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        (val_forward - Complex::new(sign, 0.0) * val_rotated).norm() < tol
    }
}
