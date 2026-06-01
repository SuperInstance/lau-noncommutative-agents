//! The spectral triple of the ENTIRE Lau ecosystem.

use nalgebra::{DMatrix, Complex};
use crate::algebra::CStarAlgebra;
use crate::hilbert::HilbertSpace;
use crate::dirac::DiracOperator;
use crate::spectral_triple::SpectralTriple;
use crate::spectral_action::SpectralAction;
use crate::nc_integral::DixmierTrace;
use crate::type_classification::{VonNeumannType, AgentTypeClassifier};

/// A component in the Lau ecosystem.
#[derive(Clone, Debug)]
pub struct LauComponent {
    pub name: String,
    pub dimension: usize,
    pub couplings: Vec<f64>,
}

/// The spectral triple of the entire Lau ecosystem.
pub struct LauSpectralTriple {
    pub components: Vec<LauComponent>,
    pub triple: SpectralTriple,
    pub vn_type: VonNeumannType,
}

impl LauSpectralTriple {
    pub fn new(components: Vec<LauComponent>) -> Self {
        let total_dim: usize = components.iter().map(|c| c.dimension).sum();
        assert!(total_dim > 0);
        let algebra = CStarAlgebra::new(total_dim);
        let hilbert = HilbertSpace::new(total_dim);
        let dirac_matrix = Self::build_dirac(&components, total_dim);
        let dirac = DiracOperator::from_real(dirac_matrix);
        let triple = SpectralTriple::new(algebra, hilbert, dirac);
        Self { components, triple, vn_type: AgentTypeClassifier::lau_ecosystem_type() }
    }

    fn build_dirac(components: &[LauComponent], total_dim: usize) -> DMatrix<f64> {
        let mut matrix = DMatrix::zeros(total_dim, total_dim);
        let mut offset = 0;
        for (i, comp) in components.iter().enumerate() {
            for k in 0..comp.dimension {
                if k + 1 < comp.dimension {
                    matrix[(offset + k, offset + k + 1)] = 1.0;
                    matrix[(offset + k + 1, offset + k)] = 1.0;
                }
            }
            let mut other_offset = 0;
            for (j, other) in components.iter().enumerate() {
                if i != j && j < comp.couplings.len() {
                    let coupling = comp.couplings[j];
                    if coupling.abs() > 1e-10 {
                        matrix[(offset, other_offset)] = coupling;
                        matrix[(other_offset, offset)] = coupling;
                    }
                }
                other_offset += other.dimension;
            }
            offset += comp.dimension;
        }
        matrix
    }

    pub fn full_ecosystem() -> Self {
        let components = vec![
            LauComponent { name: "lau-core".into(), dimension: 4, couplings: vec![0.0, 1.0, 0.8, 0.5, 0.7, 0.3, 0.6, 0.4] },
            LauComponent { name: "lau-agents".into(), dimension: 6, couplings: vec![1.0, 0.0, 0.9, 0.6, 0.8, 0.5, 0.7, 0.4] },
            LauComponent { name: "lau-noncommutative-agents".into(), dimension: 8, couplings: vec![0.8, 0.9, 0.0, 0.7, 0.9, 0.6, 0.8, 0.5] },
            LauComponent { name: "lau-quantum-agents".into(), dimension: 4, couplings: vec![0.5, 0.6, 0.7, 0.0, 0.4, 0.8, 0.5, 0.3] },
            LauComponent { name: "lau-topological-agents".into(), dimension: 6, couplings: vec![0.7, 0.8, 0.9, 0.4, 0.0, 0.7, 0.6, 0.5] },
            LauComponent { name: "lau-homotopy-agents".into(), dimension: 4, couplings: vec![0.3, 0.5, 0.6, 0.8, 0.7, 0.0, 0.9, 0.6] },
            LauComponent { name: "lau-stochastic-agents".into(), dimension: 6, couplings: vec![0.6, 0.7, 0.8, 0.5, 0.6, 0.9, 0.0, 0.8] },
            LauComponent { name: "lau-gauge-agents".into(), dimension: 4, couplings: vec![0.4, 0.4, 0.5, 0.3, 0.5, 0.6, 0.8, 0.0] },
        ];
        Self::new(components)
    }

    pub fn spectral_action(&self, lambda: f64) -> f64 { SpectralAction::with_cutoff(&self.triple, lambda) }
    pub fn heat_kernel(&self, t: f64) -> f64 { SpectralAction::heat_kernel(&self.triple, t) }
    pub fn nc_volume(&self) -> f64 { DixmierTrace::nc_integral(&self.triple, &self.triple.algebra.identity()).re }
    pub fn metric_dimension(&self) -> f64 { self.triple.metric_dimension() }
    pub fn spectrum(&self) -> Vec<f64> { self.triple.spectrum() }
    pub fn component_names(&self) -> Vec<&str> { self.components.iter().map(|c| c.name.as_str()).collect() }
    pub fn total_dimension(&self) -> usize { self.triple.hilbert.dimension }
    pub fn interaction_graph(&self) -> &DMatrix<Complex<f64>> { &self.triple.dirac.matrix }
}
