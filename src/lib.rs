//! # lau-noncommutative-agents
//!
//! Connes' noncommutative geometry for agent systems.
//!
//! The central object is the **spectral triple** `(A, H, D)`:
//! - `A` — a C*-algebra of agent observables
//! - `H` — a Hilbert space of agent states
//! - `D` — a self-adjoint Dirac operator with compact resolvent

pub mod algebra;
pub mod hilbert;
pub mod dirac;
pub mod spectral_triple;
pub mod distance;
pub mod spectral_action;
pub mod nc_integral;
pub mod k_theory;
pub mod index_pairing;
pub mod tomita_takesaki;
pub mod type_classification;
pub mod cyclic_cohomology;
pub mod chern_character;
pub mod lau_ecosystem;

pub use algebra::CStarAlgebra;
pub use hilbert::HilbertSpace;
pub use dirac::DiracOperator;
pub use spectral_triple::SpectralTriple;
pub use distance::connes_distance;
pub use spectral_action::SpectralAction;
pub use nc_integral::DixmierTrace;
pub use k_theory::{K0Group, K1Group};
pub use index_pairing::IndexPairing;
pub use tomita_takesaki::TomitaTakesaki;
pub use type_classification::VonNeumannType;
pub use cyclic_cohomology::CyclicCohomology;
pub use chern_character::ChernCharacter;
pub use lau_ecosystem::LauSpectralTriple;
