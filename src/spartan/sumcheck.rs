#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
use crate::errors::SpartanError;
use crate::spartan::polys::{
  multilinear::MultilinearPolynomial,
  univariate::{CompressedUniPoly, UniPoly},
};
use crate::traits::{Group, TranscriptEngineTrait};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub(crate) struct SumcheckProof<G: Group> {
  compressed_polys: Vec<CompressedUniPoly<G::Scalar>>,
}

impl<G: Group> SumcheckProof<G> {
  pub fn new(compressed_polys: Vec<CompressedUniPoly<G::Scalar>>) -> Self {
    Self { compressed_polys }
  }

  pub fn verify(
    &self,
    claim: G::Scalar,
    num_rounds: usize,
    degree_bound: usize,
    transcript: &mut G::TE,
  ) -> Result<(G::Scalar, Vec<G::Scalar>), SpartanError> {
    let mut e = claim;
    let mut r: Vec<G::Scalar> = Vec::new();

    // verify that there is a univariate polynomial for each round
    if self.compressed_polys.len() != num_rounds {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    for i in 0..self.compressed_polys.len() {
      let poly = self.compressed_polys[i].decompress(&e);

      // verify degree bound
      if poly.degree() != degree_bound {
        return Err(SpartanError::InvalidSumcheckProof);
      }

      // we do not need to check if poly(0) + poly(1) = e, as
      // decompress() call above already ensures that holds
      debug_assert_eq!(poly.eval_at_zero() + poly.eval_at_one(), e);

      // append the prover's message to the transcript
      transcript.absorb(b"p", &poly);

      //derive the verifier's challenge for the next round
      let r_i = transcript.squeeze(b"c")?;

      r.push(r_i);

      // evaluate the claimed degree-ell polynomial at r_i
      e = poly.evaluate(&r_i);
    }

    Ok((e, r))
  }

  #[inline]
  #[tracing::instrument(skip_all, name = "Spartan2::sumcheck::compute_eval_points_quadratic")]
  pub(in crate::spartan) fn compute_eval_points_quadratic<F>(
    poly_A: &MultilinearPolynomial<G::Scalar>,
    poly_B: &MultilinearPolynomial<G::Scalar>,
    comb_func: &F,
  ) -> (G::Scalar, G::Scalar)
  where
    F: Fn(&G::Scalar, &G::Scalar) -> G::Scalar + Sync,
  {
    let len = poly_A.len() / 2;
    (0..len)
      .into_par_iter()
      .map(|i| {
        // eval 0: bound_func is A(low)
        let eval_point_0 = comb_func(&poly_A[i], &poly_B[i]);

        // eval 2: bound_func is -A(low) + 2*A(high)
        let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
        let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
        let eval_point_2 = comb_func(&poly_A_bound_point, &poly_B_bound_point);
        (eval_point_0, eval_point_2)
      })
      .reduce(
        || (G::Scalar::ZERO, G::Scalar::ZERO),
        |a, b| (a.0 + b.0, a.1 + b.1),
      )
  }

  #[tracing::instrument(skip_all, name = "Spartan2::sumcheck::prove_quad_unrolled")]
  // A fork of `prove_quad` with the 0th round unrolled from the rest of the
  // for loop. This allows us to pass in `W` and `X` as references instead of
  // passing them in as a single `MultilinearPolynomial`, which would require
  // an expensive concatenation. We defer the actual instantation of a
  // `MultilinearPolynomial` to the end of the 0th round.
  pub fn prove_quad_unrolled<F>(
    claim: &G::Scalar,
    num_rounds: usize,
    poly_A: &mut MultilinearPolynomial<G::Scalar>,
    W: &Vec<G::Scalar>,
    X: &Vec<G::Scalar>,
    comb_func: F,
    transcript: &mut G::TE,
  ) -> Result<(Self, Vec<G::Scalar>, Vec<G::Scalar>), SpartanError>
  where
    F: Fn(&G::Scalar, &G::Scalar) -> G::Scalar + Sync,
  {
    let mut r: Vec<G::Scalar> = Vec::with_capacity(num_rounds);
    let mut polys: Vec<CompressedUniPoly<G::Scalar>> = Vec::with_capacity(num_rounds);
    let mut claim_per_round = *claim;

    /*          Round 0 START         */

    // Simulates `poly_B` polynomial with evaluations
    //     [W, 1, X, 0, 0, ...]
    // without actually concatenating W and X, which would be expensive.
    let virtual_poly_B = |index: usize| {
      if index < W.len() {
        W[index]
      } else if index == W.len() {
        G::Scalar::ONE
      } else if index <= W.len() + X.len() {
        let x_index = index - W.len() - 1;
        X[x_index]
      } else {
        G::Scalar::ZERO
      }
    };

    let len = poly_A.len() / 2;
    let poly = {
      // A fork of:
      //     Self::compute_eval_points_quadratic(poly_A, poly_B, &comb_func);
      // that uses `virtual_poly_B`
      let (eval_point_0, eval_point_2) = (0..len)
        .into_par_iter()
        .map(|i| {
          // eval 0: bound_func is A(low)
          let eval_point_0 = comb_func(&poly_A[i], &virtual_poly_B(i));

          // eval 2: bound_func is -A(low) + 2*A(high)
          let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
          let poly_B_bound_point =
            virtual_poly_B(len + i) + virtual_poly_B(len + i) - virtual_poly_B(i);
          let eval_point_2 = comb_func(&poly_A_bound_point, &poly_B_bound_point);
          (eval_point_0, eval_point_2)
        })
        .reduce(
          || (G::Scalar::ZERO, G::Scalar::ZERO),
          |a, b| (a.0 + b.0, a.1 + b.1),
        );

      let evals = [eval_point_0, claim_per_round - eval_point_0, eval_point_2];
      UniPoly::from_evals(&evals)
    };

    // append the prover's message to the transcript
    transcript.absorb(b"p", &poly);

    //derive the verifier's challenge for the next round
    let r_i = transcript.squeeze(b"c")?;
    r.push(r_i);
    polys.push(poly.compress());

    // Set up next round
    claim_per_round = poly.evaluate(&r_i);

    // bound all tables to the verifier's challenge
    let (_, mut poly_B) = rayon::join(
      || poly_A.bound_poly_var_top_zero_optimized(&r_i),
      || {
        // Simulates `poly_B.bound_poly_var_top(&r_i)`
        // We need to do this because we don't actually have
        // a `MultilinearPolynomial` instance for `poly_B` yet,
        // only the constituents of its (Lagrange basis) coefficients
        // `W` and `X`.
        let zero = G::Scalar::ZERO;
        let one = [G::Scalar::ONE];
        let Z_iter = W
          .par_iter()
          .chain(one.par_iter())
          .chain(X.par_iter())
          .chain(rayon::iter::repeatn(&zero, len));
        let left_iter = Z_iter.clone().take(len);
        let right_iter = Z_iter.skip(len).take(len);
        let B = left_iter
          .zip(right_iter)
          .map(|(a, b)| if *a == *b { *a } else { *a + r_i * (*b - *a) })
          .collect();
        MultilinearPolynomial::new(B)
      },
    );

    /*          Round 0 END          */

    for _ in 1..num_rounds {
      let poly = {
        let (eval_point_0, eval_point_2) =
          Self::compute_eval_points_quadratic(poly_A, &poly_B, &comb_func);

        let evals = [eval_point_0, claim_per_round - eval_point_0, eval_point_2];
        UniPoly::from_evals(&evals)
      };

      // append the prover's message to the transcript
      transcript.absorb(b"p", &poly);

      //derive the verifier's challenge for the next round
      let r_i = transcript.squeeze(b"c")?;
      r.push(r_i);
      polys.push(poly.compress());

      // Set up next round
      claim_per_round = poly.evaluate(&r_i);

      // bound all tables to the verifier's challenege
      rayon::join(
        || poly_A.bound_poly_var_top_zero_optimized(&r_i),
        || poly_B.bound_poly_var_top_zero_optimized(&r_i),
      );
    }

    let evals = vec![poly_A[0], poly_B[0]];
    std::thread::spawn(|| drop(poly_B));

    Ok((
      SumcheckProof {
        compressed_polys: polys,
      },
      r,
      evals
    ))
  }

  #[tracing::instrument(skip_all, name = "Spartan2::sumcheck::prove_quad")]
  pub fn prove_quad<F>(
    claim: &G::Scalar,
    num_rounds: usize,
    poly_A: &mut MultilinearPolynomial<G::Scalar>,
    poly_B: &mut MultilinearPolynomial<G::Scalar>,
    comb_func: F,
    transcript: &mut G::TE,
  ) -> Result<(Self, Vec<G::Scalar>, Vec<G::Scalar>), SpartanError>
  where
    F: Fn(&G::Scalar, &G::Scalar) -> G::Scalar + Sync,
  {
    let mut r: Vec<G::Scalar> = Vec::new();
    let mut polys: Vec<CompressedUniPoly<G::Scalar>> = Vec::new();
    let mut claim_per_round = *claim;
    for _ in 0..num_rounds {
      let poly = {
        let (eval_point_0, eval_point_2) =
          Self::compute_eval_points_quadratic(poly_A, poly_B, &comb_func);

        let evals = vec![eval_point_0, claim_per_round - eval_point_0, eval_point_2];
        UniPoly::from_evals(&evals)
      };

      // append the prover's message to the transcript
      transcript.absorb(b"p", &poly);

      //derive the verifier's challenge for the next round
      let r_i = transcript.squeeze(b"c")?;
      r.push(r_i);
      polys.push(poly.compress());

      // Set up next round
      claim_per_round = poly.evaluate(&r_i);

      // bound all tables to the verifier's challenege
      rayon::join(
        || poly_A.bound_poly_var_top(&r_i),
        || poly_B.bound_poly_var_top(&r_i),
      );
    }

    Ok((
      SumcheckProof {
        compressed_polys: polys,
      },
      r,
      vec![poly_A[0], poly_B[0]],
    ))
  }

  #[tracing::instrument(skip_all, name = "Spartan2::sumcheck::prove_quad_batch")]
  pub fn prove_quad_batch<F>(
    claim: &G::Scalar,
    num_rounds: usize,
    poly_A_vec: &mut Vec<MultilinearPolynomial<G::Scalar>>,
    poly_B_vec: &mut Vec<MultilinearPolynomial<G::Scalar>>,
    coeffs: &[G::Scalar],
    comb_func: F,
    transcript: &mut G::TE,
  ) -> Result<(Self, Vec<G::Scalar>, (Vec<G::Scalar>, Vec<G::Scalar>)), SpartanError>
  where
    F: Fn(&G::Scalar, &G::Scalar) -> G::Scalar + Sync,
  {
    let mut e = *claim;
    let mut r: Vec<G::Scalar> = Vec::new();
    let mut quad_polys: Vec<CompressedUniPoly<G::Scalar>> = Vec::new();

    for _j in 0..num_rounds {
      let mut evals: Vec<(G::Scalar, G::Scalar)> = Vec::new();

      for (poly_A, poly_B) in poly_A_vec.iter().zip(poly_B_vec.iter()) {
        let (eval_point_0, eval_point_2) =
          Self::compute_eval_points_quadratic(poly_A, poly_B, &comb_func);
        evals.push((eval_point_0, eval_point_2));
      }

      let evals_combined_0 = (0..evals.len()).map(|i| evals[i].0 * coeffs[i]).sum();
      let evals_combined_2 = (0..evals.len()).map(|i| evals[i].1 * coeffs[i]).sum();

      let evals = vec![evals_combined_0, e - evals_combined_0, evals_combined_2];
      let poly = UniPoly::from_evals(&evals);

      // append the prover's message to the transcript
      transcript.absorb(b"p", &poly);

      // derive the verifier's challenge for the next round
      let r_i = transcript.squeeze(b"c")?;
      r.push(r_i);

      // bound all tables to the verifier's challenege
      for (poly_A, poly_B) in poly_A_vec.iter_mut().zip(poly_B_vec.iter_mut()) {
        poly_A.bound_poly_var_top(&r_i);
        poly_B.bound_poly_var_top(&r_i);
      }

      e = poly.evaluate(&r_i);
      quad_polys.push(poly.compress());
    }

    let poly_A_final = (0..poly_A_vec.len()).map(|i| poly_A_vec[i][0]).collect();
    let poly_B_final = (0..poly_B_vec.len()).map(|i| poly_B_vec[i][0]).collect();
    let claims_prod = (poly_A_final, poly_B_final);

    Ok((SumcheckProof::new(quad_polys), r, claims_prod))
  }

  #[inline]
  #[tracing::instrument(skip_all, name = "Spartan2::sumcheck::compute_eval_points_cubic")]
  pub(in crate::spartan) fn compute_eval_points_cubic<F>(
    poly_A: &MultilinearPolynomial<G::Scalar>,
    poly_B: &MultilinearPolynomial<G::Scalar>,
    poly_C: &MultilinearPolynomial<G::Scalar>,
    poly_D: &MultilinearPolynomial<G::Scalar>,
    comb_func: &F,
  ) -> (G::Scalar, G::Scalar, G::Scalar)
  where
    F: Fn(&G::Scalar, &G::Scalar, &G::Scalar, &G::Scalar) -> G::Scalar + Sync,
  {
    let len = poly_A.len() / 2;
    (0..len)
      .into_par_iter()
      .map(|i| {
        // eval 0: bound_func is A(low)
        let eval_point_0 = comb_func(&poly_A[i], &poly_B[i], &poly_C[i], &poly_D[i]);

        // eval 2: bound_func is -A(low) + 2*A(high)
        let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
        let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
        let poly_C_bound_point = poly_C[len + i] + poly_C[len + i] - poly_C[i];
        let poly_D_bound_point = poly_D[len + i] + poly_D[len + i] - poly_D[i];
        let eval_point_2 = comb_func(
          &poly_A_bound_point,
          &poly_B_bound_point,
          &poly_C_bound_point,
          &poly_D_bound_point,
        );

        // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
        let poly_A_bound_point = poly_A_bound_point + poly_A[len + i] - poly_A[i];
        let poly_B_bound_point = poly_B_bound_point + poly_B[len + i] - poly_B[i];
        let poly_C_bound_point = poly_C_bound_point + poly_C[len + i] - poly_C[i];
        let poly_D_bound_point = poly_D_bound_point + poly_D[len + i] - poly_D[i];
        let eval_point_3 = comb_func(
          &poly_A_bound_point,
          &poly_B_bound_point,
          &poly_C_bound_point,
          &poly_D_bound_point,
        );
        (eval_point_0, eval_point_2, eval_point_3)
      })
      .reduce(
        || (G::Scalar::ZERO, G::Scalar::ZERO, G::Scalar::ZERO),
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
      )
  }

  #[tracing::instrument(skip_all, name = "Spartan2::sumcheck::prove_cubic_with_additive_term")]
  pub fn prove_cubic_with_additive_term<F>(
    claim: &G::Scalar,
    num_rounds: usize,
    poly_A: &mut MultilinearPolynomial<G::Scalar>,
    poly_B: &mut MultilinearPolynomial<G::Scalar>,
    poly_C: &mut MultilinearPolynomial<G::Scalar>,
    poly_D: &mut MultilinearPolynomial<G::Scalar>,
    comb_func: F,
    transcript: &mut G::TE,
  ) -> Result<(Self, Vec<G::Scalar>, Vec<G::Scalar>), SpartanError>
  where
    F: Fn(&G::Scalar, &G::Scalar, &G::Scalar, &G::Scalar) -> G::Scalar + Sync,
  {
    let mut r: Vec<G::Scalar> = Vec::new();
    let mut polys: Vec<CompressedUniPoly<G::Scalar>> = Vec::new();
    let mut claim_per_round = *claim;

    for _ in 0..num_rounds {
      let poly = {
        // Make an iterator returning the contributions to the evaluations
        let (eval_point_0, eval_point_2, eval_point_3) =
          Self::compute_eval_points_cubic(poly_A, poly_B, poly_C, poly_D, &comb_func);

        let evals = vec![
          eval_point_0,
          claim_per_round - eval_point_0,
          eval_point_2,
          eval_point_3,
        ];
        UniPoly::from_evals(&evals)
      };

      // append the prover's message to the transcript
      transcript.absorb(b"p", &poly);

      //derive the verifier's challenge for the next round
      let r_i = transcript.squeeze(b"c")?;
      r.push(r_i);
      polys.push(poly.compress());

      // Set up next round
      claim_per_round = poly.evaluate(&r_i);

      // bound all tables to the verifier's challenege
      rayon::join(
        || poly_A.bound_poly_var_top(&r_i),
        || {
          rayon::join(
            || poly_B.bound_poly_var_top_zero_optimized(&r_i),
            || {
              rayon::join(
                || poly_C.bound_poly_var_top_zero_optimized(&r_i),
                || poly_D.bound_poly_var_top_zero_optimized(&r_i),
              )
            },
          )
        },
      );
    }

    Ok((
      SumcheckProof {
        compressed_polys: polys,
      },
      r,
      vec![poly_A[0], poly_B[0], poly_C[0], poly_D[0]],
    ))
  }
}
