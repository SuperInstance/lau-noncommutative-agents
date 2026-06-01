//! Dirac operator — the geometric engine.
//!
//! Self-adjoint operator encoding metric, topology, and differential structure.

use nalgebra::{DMatrix, Complex};

/// A Dirac operator on a finite-dimensional Hilbert space.
pub struct DiracOperator {
    /// Real symmetric matrix representing D (self-adjoint with real entries for simplicity).
    pub matrix_real: DMatrix<f64>,
    /// Complex matrix representation.
    pub matrix: DMatrix<Complex<f64>>,
    pub hilbert_dim: usize,
}

impl DiracOperator {
    pub fn new(matrix: DMatrix<Complex<f64>>) -> Self {
        let n = matrix.nrows();
        assert_eq!(n, matrix.ncols());
        let matrix_real = Self::to_real(&matrix);
        Self { matrix, matrix_real, hilbert_dim: n }
    }

    pub fn from_real(matrix: DMatrix<f64>) -> Self {
        let n = matrix.nrows();
        assert_eq!(n, matrix.ncols());
        let complex_matrix = matrix.map(|x| Complex::new(x, 0.0));
        Self { matrix: complex_matrix, matrix_real: matrix, hilbert_dim: n }
    }

    fn to_real(m: &DMatrix<Complex<f64>>) -> DMatrix<f64> {
        let n = m.nrows();
        let mut r = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                r[(i, j)] = m[(i, j)].re;
            }
        }
        r
    }

    pub fn is_self_adjoint(&self, tol: f64) -> bool {
        (&self.matrix - &self.matrix.adjoint()).norm() < tol
    }

    pub fn eigenvalues_real(&self) -> Vec<f64> {
        self.matrix_real.symmetric_eigenvalues().data.as_slice().to_vec()
    }

    pub fn eigenvalues_sorted(&self) -> Vec<f64> {
        let mut ev = self.eigenvalues_real();
        ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ev
    }

    pub fn commutator(&self, a: &DMatrix<Complex<f64>>) -> DMatrix<Complex<f64>> {
        &self.matrix * a - a * &self.matrix
    }

    pub fn commutator_norm(&self, a: &DMatrix<Complex<f64>>) -> f64 {
        let comm = self.commutator(a);
        let n = comm.nrows();
        let mut real_mat = DMatrix::zeros(n * 2, n * 2);
        for i in 0..n {
            for j in 0..n {
                let c = comm[(i, j)];
                real_mat[(2*i, 2*j)] = c.re;
                real_mat[(2*i, 2*j+1)] = -c.im;
                real_mat[(2*i+1, 2*j)] = c.im;
                real_mat[(2*i+1, 2*j+1)] = c.re;
            }
        }
        real_mat.singular_values()[0]
    }

    pub fn laplacian(&self) -> DMatrix<Complex<f64>> { &self.matrix * &self.matrix }

    pub fn resolvent(&self, lambda: Complex<f64>) -> Option<DMatrix<Complex<f64>>> {
        let d_minus_lambda = &self.matrix - DMatrix::from_diagonal(&nalgebra::DVector::from_element(
            self.hilbert_dim, lambda,
        ));
        d_minus_lambda.try_inverse()
    }

    pub fn spectral_gap(&self) -> f64 {
        self.eigenvalues_sorted().into_iter()
            .filter(|&λ| λ.abs() > 1e-10)
            .map(|λ| λ.abs())
            .fold(f64::INFINITY, f64::min)
    }

    pub fn metric_dimension(&self) -> f64 {
        let ev = self.eigenvalues_sorted();
        let n = ev.len() as f64;
        if n < 2.0 { return 0.0; }
        let lambda_max = ev.last().unwrap().abs().max(1e-10);
        let lambda_min = ev.iter()
            .filter(|&&λ| λ.abs() > 1e-10)
            .map(|λ| λ.abs())
            .fold(f64::INFINITY, f64::min);
        if lambda_min < 1e-10 || lambda_max < 1e-10 { return 0.0; }
        (n.ln() / (lambda_max / lambda_min).ln()).max(0.0)
    }

    pub fn two_point() -> Self {
        Self::from_real(DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]))
    }

    pub fn from_distances(distances: &[Vec<f64>]) -> Self {
        let n = distances.len();
        let mut matrix = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if i != j && distances[i][j] > 0.0 {
                    matrix[(i, j)] = 1.0 / distances[i][j];
                }
            }
        }
        Self::from_real(matrix)
    }

    pub fn tensor(&self, other: &DiracOperator) -> DiracOperator {
        DiracOperator::new(self.matrix.kronecker(&other.matrix))
    }
}

/// Builder for Dirac operators.
pub struct DiracBuilder {
    dimension: usize,
    eigenvalues: Vec<f64>,
}

impl DiracBuilder {
    pub fn new(dim: usize) -> Self { Self { dimension: dim, eigenvalues: Vec::new() } }

    pub fn with_eigenvalues(mut self, ev: Vec<f64>) -> Self {
        assert_eq!(ev.len(), self.dimension);
        self.eigenvalues = ev;
        self
    }

    pub fn build_diagonal(self) -> DiracOperator {
        let diag: Vec<Complex<f64>> = self.eigenvalues.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let matrix = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(diag));
        DiracOperator::new(matrix)
    }
}
