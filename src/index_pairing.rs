//! Index pairing: K-homology × K-theory → ℤ.

use nalgebra::{DMatrix, Complex};
use crate::spectral_triple::SpectralTriple;

/// Index pairing between K-theory and K-homology.
pub struct IndexPairing;

impl IndexPairing {
    /// Pair a projection with the Dirac operator.
    pub fn pair_k0(triple: &SpectralTriple, p: &DMatrix<Complex<f64>>) -> i64 {
        let ev = triple.dirac.eigenvalues_real();
        let n = triple.hilbert.dimension;
        let mut sign_diag = Vec::new();
        for λ in &ev {
            if λ.abs() > 1e-10 {
                sign_diag.push(Complex::new(λ.signum(), 0.0));
            } else {
                sign_diag.push(Complex::new(0.0, 0.0));
            }
        }
        let f = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(sign_diag));
        let pfp = p * &f * p;
        pfp.trace().re.round() as i64
    }

    /// Pair a unitary with the Dirac operator.
    pub fn pair_k1(_triple: &SpectralTriple, u: &DMatrix<Complex<f64>>) -> i64 {
        let det = u.determinant();
        let winding = det.arg() / (2.0 * std::f64::consts::PI);
        winding.round() as i64
    }

    /// Fredholm index.
    pub fn fredholm_index(matrix: &DMatrix<Complex<f64>>, tol: f64) -> i64 {
        let n = matrix.nrows();
        // Convert to real matrix for SVD
        let mut real_mat = DMatrix::zeros(n * 2, n * 2);
        for i in 0..n {
            for j in 0..n {
                let c = matrix[(i, j)];
                real_mat[(2*i, 2*j)] = c.re;
                real_mat[(2*i, 2*j+1)] = -c.im;
                real_mat[(2*i+1, 2*j)] = c.im;
                real_mat[(2*i+1, 2*j+1)] = c.re;
            }
        }
        let sv = real_mat.singular_values();
        let dim_ker = sv.iter().filter(|&&s| s < tol).count();
        dim_ker as i64
    }

    /// Connes-Chern character: (1/n!) Tr(γ[D,a₁][D,a₂]...[D,aₙ]).
    pub fn connes_chern_character(triple: &SpectralTriple, elements: &[DMatrix<Complex<f64>>]) -> Complex<f64> {
        let n = elements.len();
        if n == 0 { return Complex::new(0.0, 0.0); }
        let mut product = triple.dirac.commutator(&elements[0]);
        for a in &elements[1..] {
            product = product * triple.dirac.commutator(a);
        }
        let trace = product.trace();
        let factorial = (1..=n).fold(1usize, |acc, k| acc * k);
        trace / Complex::new(factorial as f64, 0.0)
    }
}
