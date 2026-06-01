//! Type classification of von Neumann algebras.

use std::fmt;

/// Von Neumann algebra type classification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VonNeumannType {
    TypeI(usize),
    TypeII1,
    TypeIIInfinity,
    TypeIII0,
    TypeIIILambda(f64),
    TypeIII1,
}

impl fmt::Display for VonNeumannType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeI(n) => write!(f, "Type I_{}", n),
            Self::TypeII1 => write!(f, "Type II₁"),
            Self::TypeIIInfinity => write!(f, "Type II_∞"),
            Self::TypeIII0 => write!(f, "Type III₀"),
            Self::TypeIIILambda(λ) => write!(f, "Type III_{{{:.3}}}", λ),
            Self::TypeIII1 => write!(f, "Type III₁"),
        }
    }
}

/// Von Neumann algebra information.
#[derive(Clone, Debug)]
pub struct VonNeumannAlgebra {
    pub vn_type: VonNeumannType,
    pub dimension: usize,
    pub is_factor: bool,
}

impl VonNeumannAlgebra {
    pub fn type_i(n: usize) -> Self {
        Self { vn_type: VonNeumannType::TypeI(n), dimension: n, is_factor: true }
    }

    pub fn has_trace(vn_type: &VonNeumannType) -> bool {
        matches!(vn_type, VonNeumannType::TypeI(_) | VonNeumannType::TypeII1 | VonNeumannType::TypeIIInfinity)
    }

    pub fn coupling_constant(&self) -> f64 {
        match self.vn_type {
            VonNeumannType::TypeI(n) => n as f64,
            VonNeumannType::TypeII1 => 1.0,
            _ => f64::INFINITY,
        }
    }

    pub fn crossed_product(&self) -> VonNeumannAlgebra {
        match self.vn_type {
            VonNeumannType::TypeII1 => VonNeumannAlgebra {
                vn_type: VonNeumannType::TypeIIInfinity,
                dimension: self.dimension,
                is_factor: self.is_factor,
            },
            _ => self.clone(),
        }
    }

    pub fn continuous_decomposition(&self) -> (VonNeumannAlgebra, f64) {
        match self.vn_type {
            VonNeumannType::TypeIII0 | VonNeumannType::TypeIIILambda(_) | VonNeumannType::TypeIII1 => {
                let dual = VonNeumannAlgebra {
                    vn_type: VonNeumannType::TypeIIInfinity,
                    dimension: self.dimension,
                    is_factor: self.is_factor,
                };
                (dual, 1.0)
            }
            _ => (self.clone(), 0.0),
        }
    }
}

/// Agent-specific type classification.
pub struct AgentTypeClassifier;

impl AgentTypeClassifier {
    pub fn classify(n_agents: usize, has_trace: bool, is_open_ended: bool) -> VonNeumannType {
        if is_open_ended { VonNeumannType::TypeIII1 }
        else if has_trace { if n_agents > 0 { VonNeumannType::TypeII1 } else { VonNeumannType::TypeI(1) } }
        else { VonNeumannType::TypeI(n_agents.max(1)) }
    }

    /// The Lau ecosystem is Type III₁.
    pub fn lau_ecosystem_type() -> VonNeumannType { VonNeumannType::TypeIII1 }
}
