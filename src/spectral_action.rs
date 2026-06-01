//! Spectral action: Tr(f(D/Λ)).

use nalgebra::Complex;
use crate::spectral_triple::SpectralTriple;

/// Spectral action functional.
pub struct SpectralAction;

impl SpectralAction {
    /// Bosonic spectral action: Tr(f(D/Λ)) = Σ f(λ_i/Λ).
    pub fn bosonic<F>(triple: &SpectralTriple, f: F, lambda: f64) -> f64
    where F: Fn(f64) -> f64 {
        triple.dirac.eigenvalues_real().iter().map(|&ev| f(ev / lambda)).sum()
    }

    /// With cutoff χ(x) = (1 - x²)⁺.
    pub fn with_cutoff(triple: &SpectralTriple, lambda: f64) -> f64 {
        Self::bosonic(triple, |x| { let v = 1.0 - x*x; if v > 0.0 { v } else { 0.0 } }, lambda)
    }

    /// Heat kernel: Tr(exp(-tD²)).
    pub fn heat_kernel(triple: &SpectralTriple, t: f64) -> f64 {
        Self::bosonic(triple, |x| (-t * x * x).exp(), 1.0)
    }

    /// Step function: Tr(θ(Λ - |D|)).
    pub fn step_function(triple: &SpectralTriple, lambda: f64) -> f64 {
        triple.dirac.eigenvalues_real().iter().filter(|&&ev| ev.abs() < lambda).count() as f64
    }

    /// Seeley-deWitt coefficient a₀ = dim(H).
    pub fn a0(triple: &SpectralTriple) -> f64 { triple.hilbert.dimension as f64 }

    /// Seeley-deWitt coefficient a₂ = Tr(D²).
    pub fn a2(triple: &SpectralTriple) -> f64 {
        triple.dirac.eigenvalues_real().iter().map(|λ| λ * λ).sum()
    }

    /// Seeley-deWitt coefficient a₄ = Tr(D⁴).
    pub fn a4(triple: &SpectralTriple) -> f64 {
        triple.dirac.eigenvalues_real().iter().map(|λ| λ.powi(4)).sum()
    }

    /// Fermionic spectral action: ⟨ψ|D|ψ⟩.
    pub fn fermionic(triple: &SpectralTriple, psi: &nalgebra::DVector<Complex<f64>>) -> Complex<f64> {
        let d_psi = &triple.dirac.matrix * psi;
        psi.dotc(&d_psi)
    }

    /// Asymptotic expansion.
    pub fn asymptotic_expansion(triple: &SpectralTriple, lambda: f64, f_coeffs: &[f64]) -> f64 {
        let a = [Self::a0(triple), Self::a2(triple), Self::a4(triple)];
        let powers = [2.0, 0.0, -2.0];
        let mut result = 0.0;
        for (i, &fc) in f_coeffs.iter().enumerate() {
            if i >= 3 { break; }
            result += fc * lambda.powf(powers[i]) * a[i];
        }
        result
    }
}
