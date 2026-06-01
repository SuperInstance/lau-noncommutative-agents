//! Noncommutative integral via the Dixmier trace.
//!
//! φ(a) = Res_{s=0} Tr(a|D|^{-s})

use nalgebra::{DMatrix, Complex};
use crate::spectral_triple::SpectralTriple;

/// Dixmier trace computation.
pub struct DixmierTrace;

impl DixmierTrace {
    /// Compute the noncommutative integral of a with respect to D.
    pub fn nc_integral(triple: &SpectralTriple, a: &DMatrix<Complex<f64>>) -> Complex<f64> {
        let n = triple.hilbert.dimension;
        if n == 0 { return Complex::new(0.0, 0.0); }
        // Compute Tr(a * |D|^{-1}) / log(n) as the Dixmier trace approximation
        let ev = triple.dirac.eigenvalues_real();
        let abs_d_inv: Vec<f64> = ev.iter().map(|&λ| {
            if λ.abs() > 1e-10 { λ.abs().recip() } else { 0.0 }
        }).collect();
        let mut result = Complex::new(0.0, 0.0);
        for i in 0..n {
            for j in 0..n {
                if i == j && i < abs_d_inv.len() {
                    result += a[(i, j)] * abs_d_inv[i];
                }
            }
        }
        let log_n = (n as f64).ln().max(1.0);
        result / Complex::new(log_n, 0.0)
    }

    /// Zeta function ζ_D(s) = Tr(|D|^{-s}).
    pub fn zeta_function(triple: &SpectralTriple, s: f64) -> f64 {
        triple.dirac.eigenvalues_real().iter()
            .filter(|&&λ| λ.abs() > 1e-10)
            .map(|λ| λ.abs().powf(-s))
            .sum()
    }

    /// Eta function η_D(s) = Σ sign(λᵢ)|λᵢ|^{-s}.
    pub fn eta_function(triple: &SpectralTriple, s: f64) -> f64 {
        triple.dirac.eigenvalues_real().iter()
            .filter(|&&λ| λ.abs() > 1e-10)
            .map(|λ| λ.signum() * λ.abs().powf(-s))
            .sum()
    }

    /// Volume form |D|^{-n}.
    pub fn volume_form(triple: &SpectralTriple) -> DMatrix<Complex<f64>> {
        let n = triple.hilbert.dimension;
        let ev = triple.dirac.eigenvalues_real();
        let diag: Vec<Complex<f64>> = ev.iter().map(|&λ| {
            if λ.abs() > 1e-10 {
                Complex::new(λ.abs().powf(-(n as f64)), 0.0)
            } else {
                Complex::new(0.0, 0.0)
            }
        }).collect();
        DMatrix::from_diagonal(&nalgebra::DVector::from_vec(diag))
    }
}
