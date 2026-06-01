//! C*-algebra of agent observables.
//!
//! A C*-algebra is a Banach *-algebra with the C*-identity ||a*a|| = ||a||^2.
//! Represented as M_n(ℂ) — the prototypical finite-dimensional C*-algebra.

use nalgebra::{DMatrix, DVector, Complex};
use std::fmt;

/// A C*-algebra M_n(ℂ) of finite-dimensional agent observables.
pub struct CStarAlgebra {
    pub dimension: usize,
}

impl CStarAlgebra {
    pub fn new(n: usize) -> Self {
        assert!(n > 0);
        Self { dimension: n }
    }

    pub fn complex() -> Self { Self::new(1) }

    pub fn vector_space_dim(&self) -> usize { self.dimension * self.dimension }

    pub fn identity(&self) -> DMatrix<Complex<f64>> {
        DMatrix::identity(self.dimension, self.dimension)
    }

    pub fn zero(&self) -> DMatrix<Complex<f64>> {
        DMatrix::zeros(self.dimension, self.dimension)
    }

    pub fn adjoint(&self, a: &DMatrix<Complex<f64>>) -> DMatrix<Complex<f64>> {
        a.adjoint()
    }

    pub fn is_self_adjoint(&self, a: &DMatrix<Complex<f64>>, tol: f64) -> bool {
        (a - &a.adjoint()).norm() < tol
    }

    pub fn is_unitary(&self, a: &DMatrix<Complex<f64>>, tol: f64) -> bool {
        let adj = a.adjoint();
        let id = self.identity();
        (&adj * a - &id).norm() < tol && (a * &adj - &id).norm() < tol
    }

    pub fn is_positive(&self, a: &DMatrix<Complex<f64>>, tol: f64) -> bool {
        if !self.is_self_adjoint(a, tol) { return false; }
        // Symmetric, check eigenvalues
        let sym = self.to_real_symmetric(a);
        let eigenvalues = sym.symmetric_eigenvalues();
        eigenvalues.iter().all(|&λ| λ >= -tol)
    }

    /// Convert self-adjoint complex matrix to real symmetric for eigenvalue computation.
    fn to_real_symmetric(&self, a: &DMatrix<Complex<f64>>) -> DMatrix<f64> {
        let n = a.nrows();
        let mut m = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = a[(i, j)].re;
            }
        }
        m
    }

    pub fn norm(&self, a: &DMatrix<Complex<f64>>) -> f64 {
        // Use real matrix SVD for singular values
        let n = a.nrows();
        let m = a.ncols();
        let mut real_mat = DMatrix::zeros(n * 2, m * 2);
        for i in 0..n {
            for j in 0..m {
                let c = a[(i, j)];
                real_mat[(2*i, 2*j)] = c.re;
                real_mat[(2*i, 2*j+1)] = -c.im;
                real_mat[(2*i+1, 2*j)] = c.im;
                real_mat[(2*i+1, 2*j+1)] = c.re;
            }
        }
        real_mat.singular_values()[0]
    }

    pub fn verify_cstar_identity(&self, a: &DMatrix<Complex<f64>>, tol: f64) -> bool {
        let adj = a.adjoint();
        let star_a = &adj * a;
        let norm_star_a = self.norm(&star_a);
        let norm_a = self.norm(a);
        (norm_star_a - norm_a * norm_a).abs() < tol * norm_a.max(1e-10)
    }

    pub fn commutator(&self, a: &DMatrix<Complex<f64>>, b: &DMatrix<Complex<f64>>) -> DMatrix<Complex<f64>> {
        a * b - b * a
    }

    pub fn spectrum(&self, a: &DMatrix<Complex<f64>>) -> Vec<f64> {
        let sym = self.to_real_symmetric(a);
        sym.symmetric_eigenvalues().data.as_slice().to_vec()
    }

    pub fn spectral_radius(&self, a: &DMatrix<Complex<f64>>) -> f64 {
        self.spectrum(a).into_iter().fold(0.0f64, |m, λ| m.max(λ.abs()))
    }

    pub fn tensor(&self, other: &CStarAlgebra) -> CStarAlgebra {
        CStarAlgebra::new(self.dimension * other.dimension)
    }

    pub fn trace(&self, a: &DMatrix<Complex<f64>>) -> Complex<f64> {
        a.trace()
    }
}

impl fmt::Display for CStarAlgebra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "M_{}(ℂ)", self.dimension)
    }
}

/// A state on the C*-algebra: φ(a) = Tr(ρa).
pub struct AlgebraState {
    pub density: DMatrix<Complex<f64>>,
}

impl AlgebraState {
    pub fn new(density: DMatrix<Complex<f64>>) -> Self {
        Self { density }
    }

    pub fn tracial(n: usize) -> Self {
        Self {
            density: DMatrix::from_diagonal(&DVector::from_element(
                n, Complex::new(1.0 / n as f64, 0.0),
            )),
        }
    }

    pub fn evaluate(&self, a: &DMatrix<Complex<f64>>) -> Complex<f64> {
        (&self.density * a).trace()
    }
}
