//! Connes' distance formula.
//!
//! d(φ, ψ) = sup { |φ(a) - ψ(a)| : ||[D, a]|| ≤ 1 }

use nalgebra::{DMatrix, DVector, Complex};
use crate::spectral_triple::SpectralTriple;
use crate::algebra::AlgebraState;

/// Compute Connes' distance between two states.
pub fn connes_distance(triple: &SpectralTriple, state1: &AlgebraState, state2: &AlgebraState, n_samples: usize) -> f64 {
    let n = triple.algebra.dimension;
    let d = &triple.dirac;
    let mut max_dist = 0.0f64;

    for seed in 0..n_samples {
        let a = random_self_adjoint(n, seed);
        let comm_norm = d.commutator_norm(&a);
        if comm_norm < 1e-15 { continue; }
        let a_normalized = a.map(|x| x / Complex::new(comm_norm, 0.0));
        let val1 = state1.evaluate(&a_normalized).re;
        let val2 = state2.evaluate(&a_normalized).re;
        let diff = (val1 - val2).abs();
        if diff > max_dist { max_dist = diff; }
    }
    max_dist
}

/// Distance between two pure states.
pub fn connes_distance_pure(triple: &SpectralTriple, psi: &DVector<Complex<f64>>, phi: &DVector<Complex<f64>>, n_samples: usize) -> f64 {
    let rho_psi = AlgebraState::new(psi * psi.adjoint());
    let rho_phi = AlgebraState::new(phi * phi.adjoint());
    connes_distance(triple, &rho_psi, &rho_phi, n_samples)
}

/// Distance matrix between a set of pure states.
pub fn distance_matrix(triple: &SpectralTriple, states: &[DVector<Complex<f64>>], n_samples: usize) -> Vec<Vec<f64>> {
    let n = states.len();
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = connes_distance_pure(triple, &states[i], &states[j], n_samples);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

/// Verify the triangle inequality.
pub fn verify_triangle_inequality(triple: &SpectralTriple, s1: &AlgebraState, s2: &AlgebraState, s3: &AlgebraState, n_samples: usize) -> bool {
    let d12 = connes_distance(triple, s1, s2, n_samples);
    let d23 = connes_distance(triple, s2, s3, n_samples);
    let d13 = connes_distance(triple, s1, s3, n_samples);
    d13 <= d12 + d23 + 1e-6
}

fn random_self_adjoint(n: usize, seed: usize) -> DMatrix<Complex<f64>> {
    let mut matrix = DMatrix::zeros(n, n);
    let mut state = seed as u64;
    for i in 0..n {
        for j in i..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let real = (state as f64 / u64::MAX as f64) * 2.0 - 1.0;
            let val = Complex::new(real, 0.0);
            matrix[(i, j)] = val;
            if i != j { matrix[(j, i)] = val; }
        }
    }
    matrix
}
