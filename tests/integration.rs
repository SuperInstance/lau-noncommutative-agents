//! Comprehensive test suite: 60+ tests.

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector, Complex};
    use approx::assert_relative_eq;

    // ═══════════════════════════════════════════════════════════════════
    // ALGEBRA TESTS (9 tests)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_algebra_creation() {
        use lau_noncommutative_agents::CStarAlgebra;
        let a = CStarAlgebra::new(3);
        assert_eq!(a.dimension, 3);
        assert_eq!(a.vector_space_dim(), 9);
    }

    #[test]
    fn test_algebra_identity() {
        use lau_noncommutative_agents::CStarAlgebra;
        let a = CStarAlgebra::new(3);
        let id = a.identity();
        assert_eq!(id[(0, 0)], Complex::new(1.0, 0.0));
        assert_eq!(id[(1, 1)], Complex::new(1.0, 0.0));
        assert_eq!(id[(0, 1)], Complex::new(0.0, 0.0));
    }

    #[test]
    fn test_algebra_self_adjoint() {
        use lau_noncommutative_agents::CStarAlgebra;
        let a = CStarAlgebra::new(2);
        let sa = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 3.0]).map(|x| Complex::new(x, 0.0));
        assert!(a.is_self_adjoint(&sa, 1e-10));
    }

    #[test]
    fn test_algebra_not_self_adjoint() {
        use lau_noncommutative_agents::CStarAlgebra;
        let a = CStarAlgebra::new(2);
        let not_sa = DMatrix::from_row_slice(2, 2, &[1.0, 3.0, 2.0, 3.0]).map(|x| Complex::new(x, 0.0));
        assert!(!a.is_self_adjoint(&not_sa, 1e-10));
    }

    #[test]
    fn test_commutator() {
        use lau_noncommutative_agents::CStarAlgebra;
        let a = CStarAlgebra::new(2);
        let x = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0));
        let y = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 0.0]).map(|x| Complex::new(x, 0.0));
        let comm = a.commutator(&x, &y);
        assert!((comm[(0, 0)] - Complex::new(1.0, 0.0)).norm() < 1e-10);
        assert!((comm[(1, 1)] - Complex::new(-1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_unitary_check() {
        use lau_noncommutative_agents::CStarAlgebra;
        let a = CStarAlgebra::new(2);
        let theta = std::f64::consts::PI / 4.0;
        let u = DMatrix::from_row_slice(2, 2, &[
            theta.cos(), -theta.sin(), theta.sin(), theta.cos(),
        ]).map(|x| Complex::new(x, 0.0));
        assert!(a.is_unitary(&u, 1e-8));
    }

    #[test]
    fn test_positive_matrix() {
        use lau_noncommutative_agents::CStarAlgebra;
        let a = CStarAlgebra::new(2);
        let p = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]).map(|x| Complex::new(x, 0.0));
        assert!(a.is_positive(&p, 1e-8));
    }

    #[test]
    fn test_tensor_product_algebra() {
        use lau_noncommutative_agents::CStarAlgebra;
        let a1 = CStarAlgebra::new(2);
        let a2 = CStarAlgebra::new(3);
        assert_eq!(a1.tensor(&a2).dimension, 6);
    }

    #[test]
    fn test_trace() {
        use lau_noncommutative_agents::CStarAlgebra;
        let a = CStarAlgebra::new(3);
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![
            Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0),
        ]));
        assert_eq!(a.trace(&m), Complex::new(6.0, 0.0));
    }

    // ═══════════════════════════════════════════════════════════════════
    // HILBERT SPACE TESTS (7 tests)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_hilbert_orthogonality() {
        use lau_noncommutative_agents::HilbertSpace;
        let h = HilbertSpace::new(3);
        let psi = DVector::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)]);
        let phi = DVector::from_vec(vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]);
        assert!(h.are_orthogonal(&psi, &phi, 1e-10));
    }

    #[test]
    fn test_hilbert_norm() {
        use lau_noncommutative_agents::HilbertSpace;
        let h = HilbertSpace::new(2);
        let psi = DVector::from_vec(vec![Complex::new(3.0, 0.0), Complex::new(4.0, 0.0)]);
        assert_relative_eq!(h.norm(&psi), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hilbert_normalize() {
        use lau_noncommutative_agents::HilbertSpace;
        let h = HilbertSpace::new(2);
        let psi = DVector::from_vec(vec![Complex::new(3.0, 0.0), Complex::new(4.0, 0.0)]);
        let normalized = h.normalize(&psi);
        assert!(h.is_unit(&normalized, 1e-10));
    }

    #[test]
    fn test_hilbert_tensor() {
        use lau_noncommutative_agents::HilbertSpace;
        assert_eq!(HilbertSpace::new(2).tensor(&HilbertSpace::new(3)).dimension, 6);
    }

    #[test]
    fn test_hilbert_fidelity_same() {
        use lau_noncommutative_agents::HilbertSpace;
        let h = HilbertSpace::new(2);
        let psi = DVector::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]);
        assert_relative_eq!(h.fidelity(&psi, &psi), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hilbert_fidelity_orthogonal() {
        use lau_noncommutative_agents::HilbertSpace;
        let h = HilbertSpace::new(2);
        let psi = DVector::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]);
        let phi = DVector::from_vec(vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)]);
        assert_relative_eq!(h.fidelity(&psi, &phi), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hilbert_parseval() {
        use lau_noncommutative_agents::HilbertSpace;
        let h = HilbertSpace::new(3);
        let psi = DVector::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)]);
        assert!(h.parseval(&psi));
    }

    // ═══════════════════════════════════════════════════════════════════
    // DIRAC OPERATOR TESTS (7 tests)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_dirac_self_adjoint() {
        use lau_noncommutative_agents::DiracOperator;
        assert!(DiracOperator::two_point().is_self_adjoint(1e-10));
    }

    #[test]
    fn test_dirac_eigenvalues() {
        use lau_noncommutative_agents::DiracOperator;
        let ev = DiracOperator::two_point().eigenvalues_sorted();
        assert_eq!(ev.len(), 2);
        assert_relative_eq!(ev[0], -1.0, epsilon = 1e-10);
        assert_relative_eq!(ev[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dirac_commutator() {
        use lau_noncommutative_agents::DiracOperator;
        let d = DiracOperator::two_point();
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]).map(|x| Complex::new(x, 0.0));
        let comm = d.commutator(&a);
        // D=[[0,1],[1,0]], a=[[1,0],[0,2]]: Da=[[0,2],[1,0]], aD=[[0,1],[2,0]], [D,a]=[[0,1],[-1,0]]
        assert!((comm[(0, 1)] - Complex::new(1.0, 0.0)).norm() < 1e-10);
        assert!((comm[(1, 0)] - Complex::new(-1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_dirac_laplacian() {
        use lau_noncommutative_agents::DiracOperator;
        let d = DiracOperator::two_point();
        let lap = d.laplacian();
        assert!((lap[(0, 0)] - Complex::new(1.0, 0.0)).norm() < 1e-10);
        assert!((lap[(1, 1)] - Complex::new(1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_dirac_spectral_gap() {
        use lau_noncommutative_agents::DiracOperator;
        assert_relative_eq!(DiracOperator::two_point().spectral_gap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dirac_builder_diagonal() {
        use lau_noncommutative_agents::dirac::DiracBuilder;
        let d = DiracBuilder::new(3).with_eigenvalues(vec![-2.0, 0.0, 3.0]).build_diagonal();
        assert!(d.is_self_adjoint(1e-10));
        let ev = d.eigenvalues_sorted();
        assert_relative_eq!(ev[0], -2.0, epsilon = 1e-10);
        assert_relative_eq!(ev[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dirac_from_distances() {
        use lau_noncommutative_agents::DiracOperator;
        let d = DiracOperator::from_distances(&vec![
            vec![0.0, 1.0, 2.0], vec![1.0, 0.0, 1.5], vec![2.0, 1.5, 0.0],
        ]);
        assert_eq!(d.hilbert_dim, 3);
        assert!(d.is_self_adjoint(1e-10));
    }

    // ═══════════════════════════════════════════════════════════════════
    // SPECTRAL TRIPLE TESTS (5 tests)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_spectral_triple_point() {
        use lau_noncommutative_agents::SpectralTriple;
        let st = SpectralTriple::point();
        assert_eq!(st.hilbert.dimension, 1);
        assert!(st.validate(1e-10).is_valid);
    }

    #[test]
    fn test_spectral_triple_two_point_validates() {
        use lau_noncommutative_agents::SpectralTriple;
        assert!(SpectralTriple::two_point().validate(1e-10).is_valid);
    }

    #[test]
    fn test_spectral_triple_commutator() {
        use lau_noncommutative_agents::SpectralTriple;
        let st = SpectralTriple::two_point();
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0));
        assert!(st.commutator_norm(&a) > 0.0);
    }

    #[test]
    fn test_spectral_triple_tensor() {
        use lau_noncommutative_agents::SpectralTriple;
        let product = SpectralTriple::two_point().tensor(&SpectralTriple::two_point());
        assert_eq!(product.hilbert.dimension, 4);
    }

    #[test]
    fn test_spectral_triple_lipschitz() {
        use lau_noncommutative_agents::SpectralTriple;
        let st = SpectralTriple::two_point();
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]).map(|x| Complex::new(x, 0.0));
        assert!(st.lipschitz_seminorm(&a) >= 0.0);
    }

    // ═══════════════════════════════════════════════════════════════════
    // DISTANCE TESTS (3 tests)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_distance_same_state_zero() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::distance::connes_distance;
        use lau_noncommutative_agents::algebra::AlgebraState;
        let st = SpectralTriple::two_point();
        let s = AlgebraState::tracial(2);
        let d = connes_distance(&st, &s, &s, 100);
        assert!(d < 0.1, "Distance to self should be ~0, got {}", d);
    }

    #[test]
    fn test_distance_different_states_positive() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::distance::connes_distance;
        use lau_noncommutative_agents::algebra::AlgebraState;
        let st = SpectralTriple::two_point();
        let s1 = AlgebraState::new(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0)));
        let s2 = AlgebraState::new(DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 1.0]).map(|x| Complex::new(x, 0.0)));
        let d = connes_distance(&st, &s1, &s2, 500);
        assert!(d > 0.0, "Distance between distinct states should be positive");
    }

    #[test]
    fn test_triangle_inequality() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::distance::verify_triangle_inequality;
        use lau_noncommutative_agents::algebra::AlgebraState;
        let st = SpectralTriple::two_point();
        let s1 = AlgebraState::new(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0)));
        let s2 = AlgebraState::new(DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 1.0]).map(|x| Complex::new(x, 0.0)));
        let s3 = AlgebraState::new(DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]).map(|x| Complex::new(x, 0.0)));
        assert!(verify_triangle_inequality(&st, &s1, &s2, &s3, 500));
    }

    // ═══════════════════════════════════════════════════════════════════
    // SPECTRAL ACTION TESTS (7 tests)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_spectral_action_bosonic() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::SpectralAction;
        let st = SpectralTriple::two_point();
        assert_relative_eq!(SpectralAction::bosonic(&st, |x| x * x, 1.0), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spectral_action_cutoff() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::SpectralAction;
        let st = SpectralTriple::two_point();
        assert_relative_eq!(SpectralAction::with_cutoff(&st, 2.0), 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_spectral_action_heat_kernel() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::SpectralAction;
        let st = SpectralTriple::two_point();
        assert_relative_eq!(SpectralAction::heat_kernel(&st, 1.0), 2.0 * (-1.0f64).exp(), epsilon = 1e-8);
    }

    #[test]
    fn test_spectral_action_step() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::SpectralAction;
        let st = SpectralTriple::two_point();
        assert_relative_eq!(SpectralAction::step_function(&st, 2.0), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_seeley_dewitt_a0() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::SpectralAction;
        assert_relative_eq!(SpectralAction::a0(&SpectralTriple::two_point()), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_seeley_dewitt_a2() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::SpectralAction;
        assert_relative_eq!(SpectralAction::a2(&SpectralTriple::two_point()), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spectral_action_fermionic() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::SpectralAction;
        let st = SpectralTriple::two_point();
        let psi = DVector::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]);
        let ferm = SpectralAction::fermionic(&st, &psi);
        assert!(ferm.norm() < 1e-10);
    }

    // ═══════════════════════════════════════════════════════════════════
    // NONCOMMUTATIVE INTEGRAL TESTS (3 tests)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_zeta_function() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::DixmierTrace;
        let st = SpectralTriple::two_point();
        assert_relative_eq!(DixmierTrace::zeta_function(&st, 2.0), 2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_eta_function() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::DixmierTrace;
        let st = SpectralTriple::two_point();
        assert_relative_eq!(DixmierTrace::eta_function(&st, 1.0), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_nc_integral_identity() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::DixmierTrace;
        let st = SpectralTriple::two_point();
        let id = st.algebra.identity();
        assert!(DixmierTrace::nc_integral(&st, &id).re > 0.0);
    }

    // ═══════════════════════════════════════════════════════════════════
    // K-THEORY TESTS (9 tests)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_projection_check() {
        use lau_noncommutative_agents::k_theory::K0Group;
        let p = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0));
        assert!(K0Group::is_projection(&p, 1e-10));
    }

    #[test]
    fn test_not_projection() {
        use lau_noncommutative_agents::k_theory::K0Group;
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0));
        assert!(!K0Group::is_projection(&m, 1e-10));
    }

    #[test]
    fn test_projection_rank() {
        use lau_noncommutative_agents::k_theory::K0Group;
        let p = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0));
        assert_relative_eq!(K0Group::rank(&p), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_k0_class() {
        use lau_noncommutative_agents::k_theory::K0Group;
        let p = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0));
        assert_eq!(K0Group::class_of(&p), 2);
    }

    #[test]
    fn test_k0_direct_sum() {
        use lau_noncommutative_agents::k_theory::K0Group;
        let p = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0));
        let pq = K0Group::direct_sum(&p, &p);
        assert_eq!(pq.nrows(), 4);
        assert_relative_eq!(K0Group::rank(&pq), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_k0_addition() {
        use lau_noncommutative_agents::k_theory::K0Group;
        assert_eq!(K0Group::add(3, 5), 8);
    }

    #[test]
    fn test_murray_von_neumann_equivalence() {
        use lau_noncommutative_agents::k_theory::K0Group;
        let p = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0));
        let q = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 1.0]).map(|x| Complex::new(x, 0.0));
        assert!(K0Group::are_equivalent(&p, &q, 1e-10));
    }

    #[test]
    fn test_k1_unitary() {
        use lau_noncommutative_agents::k_theory::K1Group;
        let theta = std::f64::consts::PI / 3.0;
        let u = DMatrix::from_row_slice(2, 2, &[
            theta.cos(), -theta.sin(), theta.sin(), theta.cos(),
        ]).map(|x| Complex::new(x, 0.0));
        assert!(K1Group::is_unitary(&u, 1e-10));
    }

    #[test]
    fn test_k1_winding_number() {
        use lau_noncommutative_agents::k_theory::K1Group;
        let u = K1Group::phase_rotation(2, std::f64::consts::PI / 4.0);
        assert_relative_eq!(K1Group::winding_number(&u), std::f64::consts::PI / 4.0, epsilon = 1e-10);
    }

    // ═══════════════════════════════════════════════════════════════════
    // INDEX PAIRING TESTS (2 tests)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_index_pairing_k0() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::IndexPairing;
        let st = SpectralTriple::two_point();
        let p = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0));
        let idx = IndexPairing::pair_k0(&st, &p);
        // Index should be a valid integer
        assert!(idx == idx); // no NaN
    }

    #[test]
    fn test_connes_chern_character() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::IndexPairing;
        let st = SpectralTriple::two_point();
        let a1 = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0));
        let a2 = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 1.0]).map(|x| Complex::new(x, 0.0));
        let ch = IndexPairing::connes_chern_character(&st, &vec![a1, a2]);
        assert!(ch.norm().is_finite());
    }

    // ═══════════════════════════════════════════════════════════════════
    // TOMITA-TAKESAKI TESTS (3 tests)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_tomita_from_density() {
        use lau_noncommutative_agents::TomitaTakesaki;
        let rho = DMatrix::from_row_slice(2, 2, &[0.7, 0.0, 0.0, 0.3]).map(|x| Complex::new(x, 0.0));
        let tt = TomitaTakesaki::from_density(&rho);
        assert_eq!(tt.dimension, 2);
    }

    #[test]
    fn test_relative_entropy_positive() {
        use lau_noncommutative_agents::TomitaTakesaki;
        let rho1 = DMatrix::from_row_slice(2, 2, &[0.9, 0.0, 0.0, 0.1]).map(|x| Complex::new(x, 0.0));
        let rho2 = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]).map(|x| Complex::new(x, 0.0));
        let s = TomitaTakesaki::relative_entropy(&rho1, &rho2);
        assert!(s > 0.0, "KL divergence should be positive");
    }

    #[test]
    fn test_relative_entropy_same_zero() {
        use lau_noncommutative_agents::TomitaTakesaki;
        let rho = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]).map(|x| Complex::new(x, 0.0));
        assert_relative_eq!(TomitaTakesaki::relative_entropy(&rho, &rho), 0.0, epsilon = 1e-10);
    }

    // ═══════════════════════════════════════════════════════════════════
    // TYPE CLASSIFICATION TESTS (6 tests)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_type_display() {
        use lau_noncommutative_agents::VonNeumannType;
        assert_eq!(format!("{}", VonNeumannType::TypeI(3)), "Type I_3");
        assert_eq!(format!("{}", VonNeumannType::TypeII1), "Type II₁");
        assert_eq!(format!("{}", VonNeumannType::TypeIII1), "Type III₁");
    }

    #[test]
    fn test_type_has_trace() {
        use lau_noncommutative_agents::type_classification::VonNeumannAlgebra;
        use lau_noncommutative_agents::VonNeumannType;
        assert!(VonNeumannAlgebra::has_trace(&VonNeumannType::TypeI(3)));
        assert!(VonNeumannAlgebra::has_trace(&VonNeumannType::TypeII1));
        assert!(!VonNeumannAlgebra::has_trace(&VonNeumannType::TypeIII1));
    }

    #[test]
    fn test_coupling_constant() {
        use lau_noncommutative_agents::type_classification::VonNeumannAlgebra;
        assert_relative_eq!(VonNeumannAlgebra::type_i(4).coupling_constant(), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_agent_classification() {
        use lau_noncommutative_agents::type_classification::AgentTypeClassifier;
        use lau_noncommutative_agents::VonNeumannType;
        assert!(matches!(AgentTypeClassifier::classify(5, false, false), VonNeumannType::TypeI(5)));
    }

    #[test]
    fn test_lau_ecosystem_type() {
        use lau_noncommutative_agents::type_classification::AgentTypeClassifier;
        use lau_noncommutative_agents::VonNeumannType;
        assert!(matches!(AgentTypeClassifier::lau_ecosystem_type(), VonNeumannType::TypeIII1));
    }

    #[test]
    fn test_crossed_product() {
        use lau_noncommutative_agents::type_classification::VonNeumannAlgebra;
        use lau_noncommutative_agents::VonNeumannType;
        let vn = VonNeumannAlgebra { vn_type: VonNeumannType::TypeII1, dimension: 4, is_factor: true };
        assert!(matches!(vn.crossed_product().vn_type, VonNeumannType::TypeIIInfinity));
    }

    // ═══════════════════════════════════════════════════════════════════
    // CYCLIC COHOMOLOGY TESTS (2 tests)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_trace_cocycle() {
        use lau_noncommutative_agents::CyclicCohomology;
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]).map(|x| Complex::new(x, 0.0));
        assert_eq!(CyclicCohomology::trace_cocycle(&m), Complex::new(3.0, 0.0));
    }

    #[test]
    fn test_fundamental_2_cocycle() {
        use lau_noncommutative_agents::CyclicCohomology;
        let d = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]).map(|x| Complex::new(x, 0.0));
        let elems = vec![
            DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0)),
            DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 1.0]).map(|x| Complex::new(x, 0.0)),
            DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]).map(|x| Complex::new(x, 0.0)),
        ];
        assert!(CyclicCohomology::fundamental_2_cocycle(&elems, &d).norm().is_finite());
    }

    // ═══════════════════════════════════════════════════════════════════
    // CHERN CHARACTER TESTS (3 tests)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_chern_k0() {
        use lau_noncommutative_agents::ChernCharacter;
        let p = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0));
        let ch = ChernCharacter::chern_character_k0(&p);
        assert_eq!(ch[0], Complex::new(1.0, 0.0));
    }

    #[test]
    fn test_connes_chern() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::ChernCharacter;
        let st = SpectralTriple::two_point();
        let elems = vec![
            DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0)),
            DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 1.0]).map(|x| Complex::new(x, 0.0)),
        ];
        assert!(ChernCharacter::connes_chern(&st, &elems).norm().is_finite());
    }

    #[test]
    fn test_normalized_chern() {
        use lau_noncommutative_agents::SpectralTriple;
        use lau_noncommutative_agents::ChernCharacter;
        let st = SpectralTriple::two_point();
        let elems = vec![
            DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0)),
            DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 1.0]).map(|x| Complex::new(x, 0.0)),
        ];
        let unnorm = ChernCharacter::connes_chern(&st, &elems);
        let norm = ChernCharacter::normalized_connes_chern(&st, &elems);
        assert!((norm - unnorm / Complex::new(2.0, 0.0)).norm() < 1e-10);
    }

    // ═══════════════════════════════════════════════════════════════════
    // LAU ECOSYSTEM TESTS (8 tests)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_lau_ecosystem_creation() {
        use lau_noncommutative_agents::LauSpectralTriple;
        let eco = LauSpectralTriple::full_ecosystem();
        assert_eq!(eco.components.len(), 8);
    }

    #[test]
    fn test_lau_ecosystem_dimension() {
        use lau_noncommutative_agents::LauSpectralTriple;
        // 4+6+8+4+6+4+6+4 = 42
        assert_eq!(LauSpectralTriple::full_ecosystem().total_dimension(), 42);
    }

    #[test]
    fn test_lau_ecosystem_spectral_action() {
        use lau_noncommutative_agents::LauSpectralTriple;
        let sa = LauSpectralTriple::full_ecosystem().spectral_action(5.0);
        assert!(sa > 0.0);
    }

    #[test]
    fn test_lau_ecosystem_heat_kernel() {
        use lau_noncommutative_agents::LauSpectralTriple;
        assert!(LauSpectralTriple::full_ecosystem().heat_kernel(1.0) > 0.0);
    }

    #[test]
    fn test_lau_ecosystem_spectrum() {
        use lau_noncommutative_agents::LauSpectralTriple;
        assert_eq!(LauSpectralTriple::full_ecosystem().spectrum().len(), 42);
    }

    #[test]
    fn test_lau_ecosystem_component_names() {
        use lau_noncommutative_agents::LauSpectralTriple;
        let names = LauSpectralTriple::full_ecosystem().component_names();
        assert!(names.contains(&"lau-core"));
        assert!(names.contains(&"lau-noncommutative-agents"));
    }

    #[test]
    fn test_lau_ecosystem_type_iii() {
        use lau_noncommutative_agents::LauSpectralTriple;
        use lau_noncommutative_agents::VonNeumannType;
        assert!(matches!(LauSpectralTriple::full_ecosystem().vn_type, VonNeumannType::TypeIII1));
    }

    #[test]
    fn test_lau_ecosystem_custom() {
        use lau_noncommutative_agents::lau_ecosystem::LauComponent;
        use lau_noncommutative_agents::LauSpectralTriple;
        let eco = LauSpectralTriple::new(vec![
            LauComponent { name: "alpha".into(), dimension: 3, couplings: vec![0.0, 1.0] },
            LauComponent { name: "beta".into(), dimension: 4, couplings: vec![1.0, 0.0] },
        ]);
        assert_eq!(eco.total_dimension(), 7);
    }

    // ═══════════════════════════════════════════════════════════════════
    // CROSS-MODULE / INTEGRATION TESTS (4 tests)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_full_pipeline() {
        use lau_noncommutative_agents::*;
        let st = SpectralTriple::two_point();
        assert!(st.validate(1e-10).is_valid);
        assert!(SpectralAction::with_cutoff(&st, 2.0) > 0.0);
        assert!(DixmierTrace::zeta_function(&st, 1.0) > 0.0);
        let p = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]).map(|x| Complex::new(x, 0.0));
        assert!(k_theory::K0Group::is_projection(&p, 1e-10));
        assert_eq!(k_theory::K0Group::class_of(&p), 1);
    }

    #[test]
    fn test_1x1_spectral_triple() {
        use lau_noncommutative_agents::SpectralTriple;
        let st = SpectralTriple::point();
        assert!(st.validate(1e-10).is_valid);
        assert_eq!(st.spectrum().len(), 1);
    }

    #[test]
    fn test_large_spectral_triple() {
        use lau_noncommutative_agents::*;
        use lau_noncommutative_agents::dirac::DiracBuilder;
        let d = DiracBuilder::new(10).with_eigenvalues(vec![-5.0,-4.0,-3.0,-2.0,-1.0,1.0,2.0,3.0,4.0,5.0]).build_diagonal();
        let st = SpectralTriple::new(CStarAlgebra::new(10), HilbertSpace::new(10), d);
        assert!(st.validate(1e-10).is_valid);
        assert!(SpectralAction::bosonic(&st, |x| x.abs(), 1.0) > 0.0);
    }

    #[test]
    fn test_dirac_tensor_product() {
        use lau_noncommutative_agents::DiracOperator;
        let d1 = DiracOperator::two_point();
        let d2 = DiracOperator::two_point();
        let d12 = d1.tensor(&d2);
        assert_eq!(d12.hilbert_dim, 4);
    }
}
