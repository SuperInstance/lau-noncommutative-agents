//! Chern character — bridge between K-theory and cyclic cohomology.

use nalgebra::{DMatrix, Complex};
use crate::spectral_triple::SpectralTriple;

/// Chern character computations.
pub struct ChernCharacter;

impl ChernCharacter {
    /// Chern character of a projection in K₀.
    pub fn chern_character_k0(p: &DMatrix<Complex<f64>>) -> Vec<Complex<f64>> {
        let mut components = Vec::new();
        components.push(p.trace());
        let p_sq = p * p;
        components.push((p - &p_sq).trace());
        components
    }

    /// Chern character of a unitary in K₁.
    pub fn chern_character_k1(u: &DMatrix<Complex<f64>>) -> Complex<f64> {
        let u_inv = u.adjoint();
        let one = DMatrix::identity(u.nrows(), u.nrows());
        (u * &u_inv - &one).trace()
    }

    /// Connes-Chern character of a spectral triple.
    pub fn connes_chern(triple: &SpectralTriple, elements: &[DMatrix<Complex<f64>>]) -> Complex<f64> {
        let n = elements.len();
        if n == 0 { return Complex::new(0.0, 0.0); }
        let mut product = elements[0].clone();
        for elem in elements.iter().take(n).skip(1) {
            product *= triple.dirac.commutator(elem);
        }
        product.trace()
    }

    /// Normalized Connes-Chern character (divided by n!).
    pub fn normalized_connes_chern(triple: &SpectralTriple, elements: &[DMatrix<Complex<f64>>]) -> Complex<f64> {
        let n = elements.len();
        let factorial: usize = (1..=n).product();
        Self::connes_chern(triple, elements) / Complex::new(factorial as f64, 0.0)
    }

    /// Pair the Chern character with cyclic cohomology.
    pub fn pair_with_cyclic(triple: &SpectralTriple, p: &DMatrix<Complex<f64>>) -> Complex<f64> {
        let n = triple.hilbert.dimension.min(4);
        let elements: Vec<_> = (0..n).map(|_| p.clone()).collect();
        Self::connes_chern(triple, &elements)
    }

    /// A-hat genus.
    pub fn a_hat_genus(triple: &SpectralTriple) -> f64 {
        triple.dirac.eigenvalues_real().iter()
            .filter(|&&λ| λ.abs() > 1e-10)
            .map(|&λ| {
                let x = λ.abs() / 2.0;
                x / x.sinh().max(1e-15)
            })
            .product::<f64>()
    }

    /// Todd class.
    pub fn todd_class(triple: &SpectralTriple) -> f64 {
        triple.dirac.eigenvalues_real().iter()
            .filter(|&&λ| λ.abs() > 1e-10)
            .map(|&λ| {
                let x = λ.abs();
                x / (1.0 - (-x).exp())
            })
            .product::<f64>()
    }
}
