# lau-noncommutative-agents

Connes' noncommutative geometry applied to agent systems — spectral triples, Dirac operators, Connes' distance, K-theory, index pairing, Tomita-Takesaki theory, cyclic cohomology, Chern characters, and a spectral model of the entire Lau ecosystem.

Built on `nalgebra` (complex linear algebra), `num-complex`, and `serde`. Every structure is grounded in the mathematics of operator algebras and noncommutative geometry.

---

## What This Does

This library implements the core machinery of Alain Connes' noncommutative geometry (NCG) and applies it to agent systems:

- **C\*-algebras** — Finite-dimensional M_n(ℂ) algebras with adjoints, norms, C\*-identity verification, spectrum, positivity checks, commutators, and tensor products.
- **Hilbert spaces** — Finite-dimensional ℂⁿ with inner products, normalization, orthogonality, projections, fidelity, density matrices, and Parseval's identity.
- **Dirac operators** — Self-adjoint operators encoding geometry: eigenvalue computation, spectral gap, metric dimension, Laplacian, resolvent, commutator norms, and a builder pattern for custom Dirac operators.
- **Spectral triples (A, H, D)** — The fundamental object of NCG. Validation, Lipschitz seminorms, tensor products, point and two-point geometries.
- **Connes' distance** — The spectral metric: d(φ, ψ) = sup{ |φ(a) − ψ(a)| : ‖[D, a]‖ ≤ 1 }. Distance matrices and triangle inequality verification.
- **Spectral action** — Tr(f(D/Λ)): bosonic action, heat kernel, step function, Seeley-deWitt coefficients (a₀, a₂, a₄), fermionic action, and asymptotic expansions.
- **Dixmier trace / NC integral** — Noncommutative integration via φ(a) = Res_{s=0} Tr(a|D|⁻ˢ), plus zeta and eta functions of D.
- **K-theory** — K₀ (projections, rank, direct sum, equivalence) and K₁ (unitaries, winding numbers, Bott periodicity).
- **Index pairing** — K-theory × K-homology → ℤ: pair projections and unitaries with Dirac operators, Fredholm index, Connes-Chern character.
- **Tomita-Takesaki theory** — Modular operators, modular automorphism groups σ_t(a) = Δⁱᵗ a Δ⁻ⁱᵗ, Tomita operator, and relative entropy S(ρ₁‖ρ₂).
- **Von Neumann type classification** — Type I, II₁, II_∞, III₀, III_λ, III₁. Agent-specific classification logic.
- **Cyclic cohomology** — Trace cocycles (HC⁰), fundamental 2-cocycles, cyclic property verification.
- **Chern characters** — Bridge K-theory to cyclic cohomology: ch₀ (projections), ch₁ (unitaries), Connes-Chern character, Â-genus, Todd class.
- **Lau ecosystem** — A spectral triple over the entire Lau agent ecosystem (8 components, 42-dimensional Hilbert space), with inter-component couplings, spectral action, heat kernel, and noncommutative volume.

---

## Key Idea

In noncommutative geometry, a "space" is described not by points and charts, but by a **spectral triple** (A, H, D):

- **A** = algebra of "functions" (agent observables)
- **H** = Hilbert space of "spinors" (agent states)
- **D** = Dirac operator encoding metric and differential structure

The geometry — distances, dimensions, integrals, topology — emerges entirely from the spectrum of D and its interactions with A. This library makes that abstract framework concrete and computational: every object is a matrix you can inspect, every invariant is a number you can compute.

---

## Install

```toml
[dependencies]
lau-noncommutative-agents = "0.1"
```

Or clone and build:

```bash
git clone https://github.com/SuperInstance/lau-noncommutative-agents.git
cd lau-noncommutative-agents
cargo build
```

**Requirements:** Rust 2021 edition (≥ 1.56).

---

## Quick Start

### Create a spectral triple and compute its spectrum

```rust
use lau_noncommutative_agents::*;

let algebra = CStarAlgebra::new(3);
let hilbert = HilbertSpace::new(3);
let dirac = DiracOperator::two_point(); // 2×2 example
// Or build a custom one:
let dirac = DiracBuilder::new(3)
    .with_eigenvalues(vec![1.0, -1.0, 2.0])
    .build_diagonal();

let triple = SpectralTriple::new(algebra, hilbert, dirac);
println!("spectrum: {:?}", triple.spectrum());
println!("metric dimension: {:.4}", triple.metric_dimension());
```

### Connes' distance between states

```rust
use lau_noncommutative_agents::*;

let triple = SpectralTriple::two_point();
let state1 = AlgebraState::tracial(2);
let state2 = AlgebraState::new(nalgebra::dmatrix![
    nalgebra::Complex::new(1.0, 0.0), nalgebra::Complex::new(0.0, 0.0);
    nalgebra::Complex::new(0.0, 0.0), nalgebra::Complex::new(0.0, 0.0)
]);

let dist = connes_distance(&triple, &state1, &state2, 100);
println!("Connes distance: {:.4}", dist);
```

### Spectral action and heat kernel

```rust
use lau_noncommutative_agents::*;

let triple = SpectralTriple::two_point();
let action = SpectralAction::with_cutoff(&triple, 2.0);
let heat = SpectralAction::heat_kernel(&triple, 0.1);
let a0 = SpectralAction::a0(&triple);
let a2 = SpectralAction::a2(&triple);

println!("spectral action (cutoff Λ=2): {:.4}", action);
println!("heat kernel Tr(exp(-0.1·D²)): {:.4}", heat);
println!("Seeley-deWitt a₀={}, a₂={:.4}", a0, a2);
```

### The full Lau ecosystem

```rust
use lau_noncommutative_agents::*;

let ecosystem = LauSpectralTriple::full_ecosystem();
println!("total dimension: {}", ecosystem.total_dimension());
println!("components: {:?}", ecosystem.component_names());
println!("metric dimension: {:.4}", ecosystem.metric_dimension());
println!("spectral action: {:.4}", ecosystem.spectral_action(5.0));
println!("heat kernel: {:.4}", ecosystem.heat_kernel(0.1));
println!("NC volume: {:.4}", ecosystem.nc_volume());
println!("von Neumann type: {}", ecosystem.vn_type);
```

---

## API Reference

### CStarAlgebra (`algebra`)

| Method | Description |
|--------|-------------|
| `CStarAlgebra::new(n)` | M_n(ℂ) algebra |
| `.identity()`, `.zero()` | Identity/zero matrix |
| `.adjoint(&a)` | Hermitian conjugate |
| `.is_self_adjoint(&a, tol)`, `.is_unitary(&a, tol)`, `.is_positive(&a, tol)` | Property checks |
| `.norm(&a)` | Operator norm via SVD |
| `.verify_cstar_identity(&a, tol)` | Check ‖a\*a‖ = ‖a‖² |
| `.commutator(&a, &b)` | [a, b] = ab − ba |
| `.spectrum(&a)`, `.spectral_radius(&a)` | Eigenvalues and spectral radius |
| `.tensor(&other)` | Tensor product algebra M_{nm}(ℂ) |
| `.trace(&a)` | Matrix trace |

**AlgebraState** — A state φ(a) = Tr(ρa) via density matrix. `AlgebraState::tracial(n)` gives the normalized trace.

### HilbertSpace (`hilbert`)

| Method | Description |
|--------|-------------|
| `HilbertSpace::new(n)` | ℂⁿ |
| `.basis(k)`, `.zero()` | Basis vectors |
| `.inner(&psi, &phi)`, `.norm(&psi)` | Inner product and norm |
| `.normalize(&psi)`, `.is_unit(&psi, tol)` | Normalization |
| `.are_orthogonal(&psi, &phi, tol)` | Orthogonality check |
| `.project(&onto, &v)` | Orthogonal projection |
| `.density_matrix(&psi)` | ρ = |ψ⟩⟨ψ| |
| `.fidelity(&psi, &phi)` | |⟨ψ|φ⟩|² |
| `.parseval(&psi)` | Verify Parseval's identity |
| `.tensor(&other)`, `.direct_sum(&other)` | Product/sum spaces |

**BoundedOperator** — Matrix operators on H with norms, adjoints, commutators.

### DiracOperator (`dirac`)

| Method | Description |
|--------|-------------|
| `DiracOperator::new(matrix)` / `::from_real(matrix)` | Construct from complex/real matrix |
| `.is_self_adjoint(tol)` | Verify D = D\* |
| `.eigenvalues_real()`, `.eigenvalues_sorted()` | Spectrum |
| `.commutator(&a)`, `.commutator_norm(&a)` | [D, a] and ‖[D, a]‖ |
| `.laplacian()` | D² |
| `.resolvent(λ)` | (D − λ)⁻¹ |
| `.spectral_gap()` | Smallest nonzero |λ| |
| `.metric_dimension()` | Weyl dimension from eigenvalue growth |
| `.tensor(&other)` | Product geometry Dirac |
| `DiracBuilder::new(dim).with_eigenvalues(ev).build_diagonal()` | Custom Dirac |

### SpectralTriple (`spectral_triple`)

| Method | Description |
|--------|-------------|
| `SpectralTriple::new(algebra, hilbert, dirac)` | Construct (A, H, D) |
| `.validate(tol)` | Check self-adjointness, compact resolvent, bounded commutator |
| `.lipschitz_seminorm(&a)` | ‖[D, a]‖ — the noncommutative Lipschitz constant |
| `.tensor(&other)` | Product geometry |
| `::point()`, `::two_point()` | Canonical examples |
| `.spectrum()`, `.metric_dimension()` | Spectral invariants |

### Connes' Distance (`distance`)

| Function | Description |
|----------|-------------|
| `connes_distance(&triple, &state1, &state2, n_samples)` | d(φ, ψ) = sup{ |φ(a)−ψ(a)| : ‖[D,a]‖ ≤ 1 } |
| `connes_distance_pure(&triple, &psi, &phi, n)` | Distance between pure states |
| `distance_matrix(&triple, &states, n)` | Full pairwise distance matrix |
| `verify_triangle_inequality(...)` | Metric space axiom check |

### SpectralAction (`spectral_action`)

| Method | Description |
|--------|-------------|
| `SpectralAction::bosonic(&triple, f, Λ)` | Tr(f(D/Λ)) for arbitrary f |
| `::with_cutoff(&triple, Λ)` | Tr((1−(D/Λ)²)⁺) |
| `::heat_kernel(&triple, t)` | Tr(exp(−tD²)) |
| `::step_function(&triple, Λ)` | #{|λ| < Λ} |
| `::a0(&triple)`, `::a2(&triple)`, `::a4(&triple)` | Seeley-deWitt coefficients |
| `::fermionic(&triple, &psi)` | ⟨ψ|D|ψ⟩ |
| `::asymptotic_expansion(&triple, Λ, &coeffs)` | Asymptotic expansion of the action |

### DixmierTrace (`nc_integral`)

| Method | Description |
|--------|-------------|
| `DixmierTrace::nc_integral(&triple, &a)` | Noncommutative integral ≈ Tr(a|D|⁻¹)/log(n) |
| `::zeta_function(&triple, s)` | ζ_D(s) = Tr(|D|⁻ˢ) |
| `::eta_function(&triple, s)` | η_D(s) = Σ sign(λᵢ)|λᵢ|⁻ˢ |
| `::volume_form(&triple)` | |D|⁻ⁿ diagonal matrix |

### K-Theory (`k_theory`)

| Method | Description |
|--------|-------------|
| `K0Group::is_projection(&p, tol)` | Verify p² = p = p\* |
| `K0Group::rank(&p)`, `::class_of(&p)` | Rank and K₀ class |
| `K0Group::direct_sum(&p, &q)` | Direct sum of projections |
| `K0Group::projection_of_rank(n, k)` | Rank-k projection in M_n |
| `K1Group::is_unitary(&u, tol)` | Verify u\*u = uu\* = I |
| `K1Group::winding_number(&u)` | arg(det(u)) |
| `K1Group::phase_rotation(n, θ)` | e^{iθ} ⊕ I_{n-1} |

### IndexPairing (`index_pairing`)

| Method | Description |
|--------|-------------|
| `IndexPairing::pair_k0(&triple, &p)` | Index pairing with K₀ projection |
| `::pair_k1(&triple, &u)` | Index pairing with K₁ unitary |
| `::fredholm_index(&matrix, tol)` | dim(ker(F)) via SVD |
| `::connes_chern_character(&triple, &elements)` | (1/n!) Tr(γ[D,a₁]…[D,aₙ]) |

### TomitaTakesaki (`tomita_takesaki`)

| Method | Description |
|--------|-------------|
| `TomitaTakesaki::from_density(&rho)` | Modular operator from density matrix |
| `::from_state(&omega)` | From a pure state |
| `.modular_automorphism(&a, t)` | σ_t(a) = Δ^{it} a Δ^{−it} |
| `.tomita_operator()` | S = JΔ^{1/2} (simplified) |
| `TomitaTakesaki::relative_entropy(&ρ₁, &ρ₂)` | S(ρ₁‖ρ₂) = Tr(ρ₁(log ρ₁ − log ρ₂)) |

### VonNeumannType / AgentTypeClassifier (`type_classification`)

| Variant | Description |
|---------|-------------|
| `TypeI(n)` | Matrix algebra M_n |
| `TypeII1` | Has unique trace, coupling constant = 1 |
| `TypeIIInfinity` | Crossed product of II₁ |
| `TypeIII0`, `TypeIIILambda(λ)`, `TypeIII1` | No trace; classified by modular spectrum |

`AgentTypeClassifier::classify(n_agents, has_trace, is_open_ended)` maps agent system properties to von Neumann types.

### CyclicCohomology (`cyclic_cohomology`)

| Method | Description |
|--------|-------------|
| `CyclicCohomology::trace_cocycle(&a)` | HC⁰: Tr(a) |
| `::fundamental_2_cocycle(&elements, &dirac)` | Tr(a₀[D,a₁][D,a₂]) |
| `::hc0(&elements)` | All traces |
| `::is_cyclic_cocycle(phi, n, elements, tol)` | Verify cyclic property |

### ChernCharacter (`chern_character`)

| Method | Description |
|--------|-------------|
| `ChernCharacter::chern_character_k0(&p)` | ch(p) = [Tr(p), Tr(p−p²), …] |
| `::chern_character_k1(&u)` | ch₁(u) = Tr(uu\* − I) |
| `::connes_chern(&triple, &elements)` | Unnormalized Connes-Chern character |
| `::a_hat_genus(&triple)` | Â-genus from Dirac spectrum |
| `::todd_class(&triple)` | Todd class from Dirac spectrum |

### LauSpectralTriple (`lau_ecosystem`)

| Method | Description |
|--------|-------------|
| `LauSpectralTriple::full_ecosystem()` | 8-component ecosystem (42-dim) |
| `LauSpectralTriple::new(components)` | Custom ecosystem from components |
| `.spectral_action(Λ)`, `.heat_kernel(t)` | Spectral invariants |
| `.nc_volume()` | Noncommutative volume |
| `.metric_dimension()`, `.spectrum()` | Geometric invariants |
| `.component_names()`, `.total_dimension()` | Structure info |
| `.interaction_graph()` | Full Dirac operator (coupling matrix) |

---

## How It Works

### Spectral Triples

A spectral triple (A, H, D) replaces the classical notion of a Riemannian spin manifold:

| Classical | Noncommutative |
|-----------|---------------|
| Smooth functions C^∞(M) | C\*-algebra A |
| Square-integrable spinors L²(S) | Hilbert space H |
| Dirac operator /∂̸ | Dirac operator D |
| Geodesic distance d(x, y) | Connes' distance d(φ, ψ) |
| Volume ∫ f dvol | Dixmier trace φ(f) |
| de Rham cohomology | Cyclic cohomology |
| Chern character | Connes-Chern character |

### Connes' Distance Formula

For two states φ, ψ on the algebra:

$$d(\varphi, \psi) = \sup\{|\varphi(a) - \psi(a)| : \|[D, a]\| \leq 1\}$$

The library estimates this by sampling random self-adjoint elements, normalizing by their commutator norm, and tracking the maximum state difference.

### Spectral Action

The bosonic spectral action counts eigenvalues of D weighted by a cutoff function:

$$\text{Tr}(f(D/\Lambda)) = \sum_i f(\lambda_i / \Lambda)$$

For the heat kernel, f(x) = exp(−tx²), giving Tr(exp(−tD²)). The Seeley-deWitt coefficients a₀ = dim(H), a₂ = Tr(D²), a₄ = Tr(D⁴) appear in the asymptotic expansion as Λ → ∞.

### K-Theory and Index Pairing

K₀(A) classifies projections (p² = p = p\*) up to equivalence; K₁(A) classifies unitaries (u\*u = uu\* = I). The index pairing maps (K-theory) × (K-homology) → ℤ via Fredholm operators associated to D.

### Tomita-Takesaki Theory

Given a state ω with density matrix ρ, the modular operator Δ = ρ generates a 1-parameter automorphism group σ_t(a) = Δ^{it} a Δ^{−it}. The relative entropy S(ρ₁‖ρ₂) = Tr(ρ₁(log ρ₁ − log ρ₂)) measures distinguishability of states.

### The Lau Ecosystem

The full Lau ecosystem is modeled as a spectral triple with 8 components (lau-core, lau-agents, lau-noncommutative-agents, lau-quantum-agents, lau-topological-agents, lau-homotopy-agents, lau-stochastic-agents, lau-gauge-agents) on a 42-dimensional Hilbert space. Each component contributes a block to the Dirac operator, with inter-block couplings encoding ecosystem interactions. The system is classified as Type III₁ (the most general von Neumann algebra type).

---

## The Math

### The C\*-Identity

$$\|a^* a\| = \|a\|^2$$

This defining property of C\*-algebras is verified numerically: `algebra.verify_cstar_identity(&a, tol)`.

### The Spectral Triple Axioms

For (A, H, D) to be a valid spectral triple:
1. D is self-adjoint: D = D\*
2. D has compact resolvent: (D − λ)⁻¹ is compact for λ ∉ spec(D)
3. [D, a] is bounded for all a ∈ A

### Metric Dimension (Weyl Asymptotics)

$$d = \lim_{\lambda \to \infty} \frac{\log N(\lambda)}{\log \lambda}$$

where N(λ) counts eigenvalues with |λ_i| ≤ λ. Estimated from the eigenvalue ratio as:

$$d \approx \frac{\log n}{\log(\lambda_{\max} / \lambda_{\min})}$$

### Dixmier Trace

$$\oint a = \text{Res}_{s=0}\, \text{Tr}(a|D|^{-s})$$

Approximated as Tr(a|D|⁻¹) / log(n) in finite dimensions.

### Connes-Chern Character

For a spectral triple with even grading γ:

$$\text{Ch}_n(a_0, \ldots, a_n) = \frac{1}{n!}\, \text{Tr}(\gamma\, [D, a_0][D, a_1] \cdots [D, a_n])$$

### Â-Genus

$$\hat{A} = \prod_{\lambda \in \text{spec}(D)} \frac{\lambda/2}{\sinh(\lambda/2)}$$

### Relative Entropy

$$S(\rho_1 \| \rho_2) = \text{Tr}(\rho_1 (\log \rho_1 - \log \rho_2))$$

---

## Tests

78 integration tests covering:

```bash
cargo test
```

- C\*-algebra operations (identity, adjoint, norm, commutators, C\*-identity, spectrum, positivity, tensor products)
- Hilbert space operations (inner products, norms, orthogonality, projections, fidelity, Parseval)
- Dirac operators (self-adjointness, eigenvalues, spectral gap, metric dimension, Laplacian, resolvent, tensor products)
- Spectral triple validation, Lipschitz seminorms, tensor products
- Connes' distance (state distances, distance matrices, triangle inequality)
- Spectral action (bosonic, heat kernel, cutoff, step function, Seeley-deWitt coefficients, fermionic)
- Dixmier trace, zeta function, eta function, volume form
- K₀ (projections, rank, equivalence, direct sum) and K₁ (unitaries, winding numbers)
- Index pairing (K₀, K₁, Fredholm index, Connes-Chern character)
- Tomita-Takesaki (modular automorphisms, relative entropy)
- Von Neumann type classification
- Cyclic cohomology (trace cocycles, 2-cocycles, cyclic property)
- Chern characters (K₀, K₁, Connes-Chern, Â-genus, Todd class)
- Lau ecosystem (construction, spectral action, heat kernel, volume, metric dimension, component structure)

---

## License

MIT
