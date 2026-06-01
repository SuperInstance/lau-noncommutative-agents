//! Spectral triple (A, H, D) — the fundamental object.

use nalgebra::{DMatrix, Complex};
use crate::algebra::CStarAlgebra;
use crate::hilbert::HilbertSpace;
use crate::dirac::DiracOperator;

/// A spectral triple (A, H, D).
pub struct SpectralTriple {
    pub algebra: CStarAlgebra,
    pub hilbert: HilbertSpace,
    pub dirac: DiracOperator,
}

/// Validation results for a spectral triple.
pub struct SpectralTripleValidation {
    pub self_adjoint: bool,
    pub compact_resolvent: bool,
    pub bounded_commutator: bool,
    pub is_valid: bool,
}

impl SpectralTriple {
    pub fn new(algebra: CStarAlgebra, hilbert: HilbertSpace, dirac: DiracOperator) -> Self {
        assert_eq!(dirac.hilbert_dim, hilbert.dimension);
        Self { algebra, hilbert, dirac }
    }

    pub fn validate(&self, tol: f64) -> SpectralTripleValidation {
        let sa = self.dirac.is_self_adjoint(tol);
        SpectralTripleValidation {
            self_adjoint: sa,
            compact_resolvent: true,
            bounded_commutator: true,
            is_valid: sa,
        }
    }

    pub fn commutator(&self, a: &DMatrix<Complex<f64>>) -> DMatrix<Complex<f64>> {
        self.dirac.commutator(a)
    }

    pub fn commutator_norm(&self, a: &DMatrix<Complex<f64>>) -> f64 {
        self.dirac.commutator_norm(a)
    }

    pub fn lipschitz_seminorm(&self, a: &DMatrix<Complex<f64>>) -> f64 {
        self.commutator_norm(a)
    }

    pub fn tensor(&self, other: &SpectralTriple) -> SpectralTriple {
        SpectralTriple::new(
            self.algebra.tensor(&other.algebra),
            self.hilbert.tensor(&other.hilbert),
            self.dirac.tensor(&other.dirac),
        )
    }

    pub fn point() -> Self {
        SpectralTriple::new(
            CStarAlgebra::complex(),
            HilbertSpace::complex(),
            DiracOperator::from_real(DMatrix::zeros(1, 1)),
        )
    }

    pub fn two_point() -> Self {
        SpectralTriple::new(
            CStarAlgebra::new(2),
            HilbertSpace::new(2),
            DiracOperator::two_point(),
        )
    }

    pub fn spectrum(&self) -> Vec<f64> { self.dirac.eigenvalues_real() }

    pub fn metric_dimension(&self) -> f64 { self.dirac.metric_dimension() }
}
