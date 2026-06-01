//! Hilbert space of agent states.

use nalgebra::{DMatrix, DVector, Complex};
use std::fmt;

/// A finite-dimensional Hilbert space.
pub struct HilbertSpace {
    pub dimension: usize,
}

impl HilbertSpace {
    pub fn new(n: usize) -> Self { assert!(n > 0); Self { dimension: n } }
    pub fn complex() -> Self { Self::new(1) }

    pub fn zero(&self) -> DVector<Complex<f64>> { DVector::zeros(self.dimension) }

    pub fn basis(&self, k: usize) -> DVector<Complex<f64>> {
        let mut v = DVector::zeros(self.dimension);
        v[k] = Complex::new(1.0, 0.0);
        v
    }

    pub fn inner(&self, psi: &DVector<Complex<f64>>, phi: &DVector<Complex<f64>>) -> Complex<f64> {
        psi.dotc(phi)
    }

    pub fn norm(&self, psi: &DVector<Complex<f64>>) -> f64 {
        self.inner(psi, psi).norm().sqrt()
    }

    pub fn normalize(&self, psi: &DVector<Complex<f64>>) -> DVector<Complex<f64>> {
        let n = self.norm(psi);
        if n < 1e-15 { return psi.clone(); }
        psi.map(|c| c / Complex::new(n, 0.0))
    }

    pub fn is_unit(&self, psi: &DVector<Complex<f64>>, tol: f64) -> bool {
        (self.norm(psi) - 1.0).abs() < tol
    }

    pub fn are_orthogonal(&self, psi: &DVector<Complex<f64>>, phi: &DVector<Complex<f64>>, tol: f64) -> bool {
        self.inner(psi, phi).norm() < tol
    }

    pub fn project(&self, onto: &DVector<Complex<f64>>, v: &DVector<Complex<f64>>) -> DVector<Complex<f64>> {
        let coeff = self.inner(onto, v) / self.inner(onto, onto);
        onto.map(|c| c * coeff)
    }

    pub fn tensor(&self, other: &HilbertSpace) -> HilbertSpace { HilbertSpace::new(self.dimension * other.dimension) }
    pub fn direct_sum(&self, other: &HilbertSpace) -> HilbertSpace { HilbertSpace::new(self.dimension + other.dimension) }

    pub fn density_matrix(&self, psi: &DVector<Complex<f64>>) -> DMatrix<Complex<f64>> {
        psi * psi.adjoint()
    }

    pub fn fidelity(&self, psi: &DVector<Complex<f64>>, phi: &DVector<Complex<f64>>) -> f64 {
        self.inner(psi, phi).norm().powi(2)
    }

    pub fn parseval(&self, psi: &DVector<Complex<f64>>) -> bool {
        let mut sum = 0.0;
        for k in 0..self.dimension {
            sum += self.inner(&self.basis(k), psi).norm().powi(2);
        }
        (sum - self.inner(psi, psi).re).abs() < 1e-10
    }
}

impl fmt::Display for HilbertSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "ℂ^{}", self.dimension) }
}

/// A bounded operator on the Hilbert space.
pub struct BoundedOperator {
    pub matrix: DMatrix<Complex<f64>>,
}

impl BoundedOperator {
    pub fn new(matrix: DMatrix<Complex<f64>>) -> Self { Self { matrix } }
    pub fn identity(n: usize) -> Self { Self::new(DMatrix::identity(n, n)) }
    pub fn zero(n: usize) -> Self { Self::new(DMatrix::zeros(n, n)) }

    pub fn apply(&self, v: &DVector<Complex<f64>>) -> DVector<Complex<f64>> { &self.matrix * v }

    pub fn norm(&self) -> f64 {
        let n = self.matrix.nrows();
        let m = self.matrix.ncols();
        let mut real_mat = DMatrix::zeros(n * 2, m * 2);
        for i in 0..n {
            for j in 0..m {
                let c = self.matrix[(i, j)];
                real_mat[(2*i, 2*j)] = c.re;
                real_mat[(2*i, 2*j+1)] = -c.im;
                real_mat[(2*i+1, 2*j)] = c.im;
                real_mat[(2*i+1, 2*j+1)] = c.re;
            }
        }
        real_mat.singular_values()[0]
    }

    pub fn adjoint(&self) -> BoundedOperator { BoundedOperator::new(self.matrix.adjoint()) }

    pub fn is_self_adjoint(&self, tol: f64) -> bool {
        (&self.matrix - &self.matrix.adjoint()).norm() < tol
    }

    pub fn trace(&self) -> Complex<f64> { self.matrix.trace() }

    pub fn commutator(&self, other: &BoundedOperator) -> BoundedOperator {
        BoundedOperator::new(&self.matrix * &other.matrix - &other.matrix * &self.matrix)
    }
}
