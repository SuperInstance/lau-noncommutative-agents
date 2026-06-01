//! K-theory: K₀(A) and K₁(A).

use nalgebra::{DMatrix, Complex};
use crate::algebra::CStarAlgebra;

/// K₀ group: equivalence classes of projections.
pub struct K0Group { pub algebra_dim: usize }

impl K0Group {
    pub fn new(algebra: &CStarAlgebra) -> Self { Self { algebra_dim: algebra.dimension } }

    pub fn is_projection(p: &DMatrix<Complex<f64>>, tol: f64) -> bool {
        let p_sq = p * p;
        let adj = p.adjoint();
        (p - &p_sq).norm() < tol && (p - adj).norm() < tol
    }

    pub fn rank(p: &DMatrix<Complex<f64>>) -> f64 { p.trace().re }

    pub fn class_of(p: &DMatrix<Complex<f64>>) -> i64 { Self::rank(p).round() as i64 }

    pub fn direct_sum(p: &DMatrix<Complex<f64>>, q: &DMatrix<Complex<f64>>) -> DMatrix<Complex<f64>> {
        let np = p.nrows();
        let nq = q.nrows();
        let mut result = DMatrix::zeros(np + nq, np + nq);
        result.view_mut((0, 0), (np, np)).copy_from(p);
        result.view_mut((np, np), (nq, nq)).copy_from(q);
        result
    }

    pub fn are_equivalent(p: &DMatrix<Complex<f64>>, q: &DMatrix<Complex<f64>>, tol: f64) -> bool {
        (Self::rank(p) - Self::rank(q)).abs() < tol
    }

    pub fn projection_of_rank(n: usize, k: usize) -> DMatrix<Complex<f64>> {
        let mut p = DMatrix::zeros(n, n);
        for i in 0..k.min(n) { p[(i, i)] = Complex::new(1.0, 0.0); }
        p
    }

    pub fn add(class_p: i64, class_q: i64) -> i64 { class_p + class_q }
    pub fn sub(class_p: i64, class_q: i64) -> i64 { class_p - class_q }
}

/// K₁ group: equivalence classes of unitaries.
pub struct K1Group { pub algebra_dim: usize }

impl K1Group {
    pub fn new(algebra: &CStarAlgebra) -> Self { Self { algebra_dim: algebra.dimension } }

    pub fn is_unitary(u: &DMatrix<Complex<f64>>, tol: f64) -> bool {
        let adj = u.adjoint();
        let id = DMatrix::identity(u.nrows(), u.nrows());
        (&adj * u - &id).norm() < tol && (u * &adj - &id).norm() < tol
    }

    pub fn winding_number(u: &DMatrix<Complex<f64>>) -> f64 { u.determinant().arg() }

    pub fn phase_rotation(n: usize, theta: f64) -> DMatrix<Complex<f64>> {
        let mut u = DMatrix::identity(n, n);
        u[(0, 0)] = Complex::new(theta.cos(), theta.sin());
        u
    }

    /// K₁(M_n(ℂ)) = 0.
    pub fn is_trivial(&self) -> bool { true }

    /// Bott periodicity.
    pub fn bott_periodicity(k0: i64, k1: i64) -> (i64, i64) { (k0, k1) }
}
