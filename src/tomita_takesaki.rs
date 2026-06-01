//! Tomita-Takesaki theory: modular structure of von Neumann algebras.

use nalgebra::{DMatrix, DVector, Complex};

/// Tomita-Takesaki modular structure.
pub struct TomitaTakesaki {
    pub modular_operator: DMatrix<Complex<f64>>,
    pub dimension: usize,
}

impl TomitaTakesaki {
    pub fn from_density(rho: &DMatrix<Complex<f64>>) -> Self {
        let n = rho.nrows();
        Self { modular_operator: rho.clone(), dimension: n }
    }

    pub fn from_state(omega: &DVector<Complex<f64>>) -> Self {
        let rho = omega * omega.adjoint();
        Self::from_density(&rho)
    }

    /// Modular automorphism group: σ_t(a) = Δ^{it} a Δ^{-it}.
    pub fn modular_automorphism(&self, a: &DMatrix<Complex<f64>>, t: f64) -> DMatrix<Complex<f64>> {
        let n = self.dimension;
        let ev = self.eigenvalues_of_modular();
        let mut delta_it = DMatrix::zeros(n, n);
        let mut delta_minus_it = DMatrix::zeros(n, n);
        for i in 0..n {
            let λ = ev[i].max(1e-15);
            let phase = Complex::new(0.0, t * λ.ln());
            delta_it[(i, i)] = phase.exp();
            delta_minus_it[(i, i)] = (-phase).exp();
        }
        &delta_it * a * &delta_minus_it
    }

    fn eigenvalues_of_modular(&self) -> Vec<f64> {
        let _n = self.dimension;
        let real = self.to_real_symmetric();
        real.symmetric_eigenvalues().data.as_slice().to_vec()
    }

    fn to_real_symmetric(&self) -> DMatrix<f64> {
        let n = self.dimension;
        let mut m = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = self.modular_operator[(i, j)].re;
            }
        }
        m
    }

    /// Tomita operator S = JΔ^{1/2} (simplified).
    pub fn tomita_operator(&self) -> DMatrix<Complex<f64>> {
        self.modular_operator.map(|x| {
            if x.re > 0.0 { Complex::new(x.re.sqrt(), 0.0) }
            else { Complex::new(0.0, 0.0) }
        })
    }

    /// Relative entropy S(ρ₁||ρ₂) = Tr(ρ₁(log ρ₁ - log ρ₂)).
    pub fn relative_entropy(rho1: &DMatrix<Complex<f64>>, rho2: &DMatrix<Complex<f64>>) -> f64 {
        let n = rho1.nrows();
        // Get eigenvalues from real symmetric matrices
        let r1 = Self::matrix_to_real(rho1);
        let r2 = Self::matrix_to_real(rho2);
        let ev1 = r1.symmetric_eigenvalues();
        let ev2 = r2.symmetric_eigenvalues();

        let mut entropy = 0.0;
        for i in 0..n {
            let p = ev1[i].max(1e-15);
            let q = ev2[i].max(1e-15);
            if p > 1e-15 { entropy += p * (p.ln() - q.ln()); }
        }
        entropy
    }

    fn matrix_to_real(m: &DMatrix<Complex<f64>>) -> DMatrix<f64> {
        let n = m.nrows();
        let mut r = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                r[(i, j)] = m[(i, j)].re;
            }
        }
        r
    }
}
