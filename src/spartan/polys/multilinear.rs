//! Main components:
//! - `MultilinearPolynomial`: Dense representation of multilinear polynomials, represented by evaluations over all possible binary inputs.
//! - `SparsePolynomial`: Efficient representation of sparse multilinear polynomials, storing only non-zero evaluations.

use std::ops::{Add, Index};

use ff::PrimeField;
use rayon::{
  prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
  },
  slice::ParallelSlice,
};
use serde::{Deserialize, Serialize};

use crate::spartan::{math::Math, polys::eq::EqPolynomial};

/// A multilinear extension of a polynomial $Z(\cdot)$, denote it as $\tilde{Z}(x_1, ..., x_m)$
/// where the degree of each variable is at most one.
///
/// This is the dense representation of a multilinear poynomial.
/// Let it be $\mathbb{G}(\cdot): \mathbb{F}^m \rightarrow \mathbb{F}$, it can be represented uniquely by the list of
/// evaluations of $\mathbb{G}(\cdot)$ over the Boolean hypercube $\{0, 1\}^m$.
///
/// For example, a 3 variables multilinear polynomial can be represented by evaluation
/// at points $[0, 2^3-1]$.
///
/// The implementation follows
/// $$
/// \tilde{Z}(x_1, ..., x_m) = \sum_{e\in {0,1}^m}Z(e) \cdot \prod_{i=1}^m(x_i \cdot e_i + (1-x_i) \cdot (1-e_i))
/// $$
///
/// Vector $Z$ indicates $Z(e)$ where $e$ ranges from $0$ to $2^m-1$.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultilinearPolynomial<Scalar: PrimeField> {
  num_vars: usize,           // the number of variables in the multilinear polynomial
  pub(crate) Z: Vec<Scalar>, // evaluations of the polynomial in all the 2^num_vars Boolean inputs
}

impl<Scalar: PrimeField> MultilinearPolynomial<Scalar> {
  /// Creates a new `MultilinearPolynomial` from the given evaluations.
  ///
  /// The number of evaluations must be a power of two.
  pub fn new(Z: Vec<Scalar>) -> Self {
    assert_eq!(Z.len(), (2_usize).pow((Z.len() as f64).log2() as u32));
    MultilinearPolynomial {
      num_vars: usize::try_from(Z.len().ilog2()).unwrap(),
      Z,
    }
  }

  /// Returns the number of variables in the multilinear polynomial
  pub const fn get_num_vars(&self) -> usize {
    self.num_vars
  }

  /// Returns the total number of evaluations.
  pub fn len(&self) -> usize {
    self.Z.len()
  }

  /// Checks if the multilinear polynomial is empty.
  ///
  /// This method returns true if the polynomial has no evaluations, and false otherwise.
  pub fn is_empty(&self) -> bool {
    self.Z.is_empty()
  }

  /// Bounds the polynomial's top variable using the given scalar.
  ///
  /// This operation modifies the polynomial in-place.
  #[tracing::instrument(skip_all, name = "MultilinearPolynomial::bound_poly_var_top")]
  pub fn bound_poly_var_top(&mut self, r: &Scalar) {
    let n = self.len() / 2;

    let (left, right) = self.Z.split_at_mut(n);

    left
      .par_iter_mut()
      .zip(right.par_iter())
      .for_each(|(a, b)| {
        *a += *r * (*b - *a);
      });

    self.Z.resize(n, Scalar::ZERO);
    self.num_vars -= 1;
  }

  /// TODO: Documentation
  #[tracing::instrument(skip_all, name = "MultilinearPolynomial::bound_poly_var_top")]
  pub fn bound_poly_var_bot(&mut self, r: &Scalar) {
    let n = self.len() / 2;
    for i in 0..n {
        self.Z[i] = self.Z[2 * i] + *r * (self.Z[2 * i + 1] - self.Z[2 * i]);
    }
    self.num_vars -= 1;
    self.Z.resize(n, Scalar::ZERO);
  }

  /// TODO: Documentation
  #[tracing::instrument(skip_all, name = "MultilinearPolynomial::bound_poly_var_top")]
  pub fn bound_poly_var_bot_zero_optimized(&mut self, r: &Scalar) {
    let n = self.len() / 2;
    for i in 0..n {
      let low = self.Z[2 * i];
      let high = self.Z[2 * i + 1];
      let low_zero = low == Scalar::ZERO;
      let high_zero = high == Scalar::ZERO;

      if low_zero && high_zero {
        self.Z[i] = Scalar::ZERO;
      } else if low_zero {
        self.Z[i] = *r * high;
      } else if high_zero {
        self.Z[i] = low + *r * (- high);
      } else {
        self.Z[i] = low + *r * (high - low);
      }
    }
    self.num_vars -= 1;
    self.Z.resize(n, Scalar::ZERO);
  }

  /// Bounds the polynomial's most significant index bit to 'r' optimized for a 
  /// high P(eval = 0).
  #[tracing::instrument(skip_all)]
  pub fn bound_poly_var_top_zero_optimized(&mut self, r: &Scalar) {
    let n = self.len() / 2;

    let (left, right) = self.Z.split_at_mut(n);
    let (right, _) = right.split_at(n);

    left
      .par_iter_mut()
      .zip(right.par_iter())
      .for_each(|(a, b)| {
        if !(*a == Scalar::ZERO && *b == Scalar::ZERO) {
          *a += *r * (*b - *a);
        }
      });

    self.Z.resize(n, Scalar::ZERO);
    self.num_vars -= 1;
  }

  /// Evaluates the polynomial at the given point.
  /// Returns Z(r) in O(n) time.
  ///
  /// The point must have a value for each variable.
  #[tracing::instrument(skip_all, name = "MultilinearPolynomial::evaluate")]
  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    // r must have a value for each variable
    assert_eq!(r.len(), self.get_num_vars());
    let chis = EqPolynomial::new(r.to_vec()).evals();
    assert_eq!(chis.len(), self.Z.len());

    (0..chis.len())
      .into_par_iter()
      .map(|i| chis[i] * self.Z[i])
      .sum()
  }

  /// Evaluates the polynomial with the given evaluations and point.
  pub fn evaluate_with(Z: &[Scalar], r: &[Scalar]) -> Scalar {
    EqPolynomial::new(r.to_vec())
      .evals()
      .into_par_iter()
      .zip(Z.into_par_iter())
      .map(|(a, b)| a * b)
      .sum()
  }

  /// Evaluates polynomial given lagrange basis
  #[tracing::instrument(skip_all, name = "MultilinearPolynomial::evaluate_with_chi")]
  pub fn evaluate_with_chi(&self, chis: &[Scalar]) -> Scalar {
    (0..chis.len())
      .into_par_iter()
      .map(|i| chis[i] * self.Z[i])
      .sum()
  }

  /// Multiplies the polynomial by a scalar.
  pub fn scalar_mul(&self, scalar: &Scalar) -> Self {
    let mut new_poly = self.clone();
    for z in &mut new_poly.Z {
      *z *= scalar;
    }
    new_poly
  }

  /// Returns the evaluations of the polynomial.
  pub fn get_Z(&self) -> &[Scalar] {
    &self.Z
  }

  /// Bounds the polynomial's top variables using the given scalars.
  #[tracing::instrument(skip_all, name = "MultilinearPolynomial::bound")]
  pub fn bound(Z: &[Scalar], L: &[Scalar]) -> Vec<Scalar> {
    let (_left_num_vars, right_num_vars) =
      EqPolynomial::<Scalar>::compute_factored_lens(Z.len().ilog2() as usize);
    let R_size = (2_usize).pow(right_num_vars as u32);

    Z
      .par_chunks(R_size)
      .enumerate()
      // TODO(moodlezoup): optimize for 0/1
      .map(|(i, row)| row.iter().map(|x| L[i] * x).collect::<Vec<Scalar>>())
      .reduce(
        || vec![Scalar::ZERO; R_size],
        |mut acc: Vec<_>, row| {
          acc.iter_mut().zip(row).for_each(|(x, y)| *x += y);
          acc
        },
      )
  }
}

impl<Scalar: PrimeField> Index<usize> for MultilinearPolynomial<Scalar> {
  type Output = Scalar;

  #[inline(always)]
  fn index(&self, index: usize) -> &Scalar {
    &(self.Z[index])
  }
}

// SPARSITY TODO:
// - ParChunkIndices: Store some power of 2 worth of Dense indices (16 / 32 / 64) in order to allow parallelism
// - DualSparseIter – takes 2 mutable slice references
// - Issue now: I can read in correct chunks, but can't write in correct chunks on binding

/// Sparse multilinear polynomial, which means the $Z(\cdot)$ is zero at most points.
/// So we do not have to store every evaluations of $Z(\cdot)$, only store the non-zero points.
///
/// For example, the evaluations are [0, 0, 0, 1, 0, 1, 0, 2].
/// The sparse polynomial only store the non-zero values, [(3, 1), (5, 1), (7, 2)].
/// In the tuple, the first is index, the second is value.
#[derive(Debug, Clone)]
pub struct SparsePolynomial<Scalar: PrimeField> {
  /// 
  pub num_vars: usize,
  ///
  pub Z: Vec<(usize, Scalar)>
}

impl<Scalar: PrimeField> SparsePolynomial<Scalar> {
  /// TODO:Documentation.
  pub fn new(num_vars: usize, Z: Vec<(usize, Scalar)>) -> Self {
    SparsePolynomial { num_vars, Z }
  }

  /// Computes the $\tilde{eq}$ extension polynomial.
  /// return 1 when a == r, otherwise return 0.
  fn compute_chi(a: &[bool], r: &[Scalar]) -> Scalar {
    assert_eq!(a.len(), r.len());
    let mut chi_i = Scalar::ONE;
    for j in 0..r.len() {
      if a[j] {
        chi_i *= r[j];
      } else {
        chi_i *= Scalar::ONE - r[j];
      }
    }
    chi_i
  }

  /// Takes O(n log n)
  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    assert_eq!(self.num_vars, r.len());

    (0..self.Z.len())
      .into_par_iter()
      .map(|i| {
        let bits = (self.Z[i].0).get_bits(r.len());
        SparsePolynomial::compute_chi(&bits, r) * self.Z[i].1
      })
      .sum()
  }

  /// TODO: Documentation
  #[tracing::instrument(skip_all, name = "SparsePolynomial::bound_poly_var_bot")]
  pub fn bound_poly_var_bot(&mut self, r: &Scalar) {
    let mut sparse_read_index = 0;
    let mut sparse_write_index = 0;
    while sparse_read_index < self.Z.len() {
      let a = self.Z[sparse_read_index];

      // Case where both low, high are non-sparse.
      if sparse_read_index != self.Z.len() - 1 
          && a.0 == self.Z[sparse_read_index+1].0 - 1  
          && a.0 % 2 == 0 {
        let b = self.Z[sparse_read_index+1];

        self.Z[sparse_write_index] = (a.0/ 2, a.1 + *r * (b.1 - a.1));
        sparse_read_index += 2;
        sparse_write_index += 1;
      } else {

        if a.0 % 2 == 0 { // low
          self.Z[sparse_write_index] = (a.0 / 2, a.1 + *r * (-a.1));
        } else { // high
          self.Z[sparse_write_index] = (a.0 / 2, *r * a.1);
        }

        sparse_read_index += 1;
        sparse_write_index += 1;
      }

    }
    self.num_vars -= 1;
    self.Z.truncate(sparse_write_index);
  }

  /// TODO: Document.
  pub fn to_dense(self) -> MultilinearPolynomial<Scalar> {
    let total_entries = 1 << self.num_vars;
    let mut evals = vec![Scalar::ZERO; total_entries];

    for sparse_entry in self.Z {
      evals[sparse_entry.0] = sparse_entry.1;
    }

    MultilinearPolynomial::new(evals)
  }

  /// TODO: Document.
  pub fn from_dense(dense: MultilinearPolynomial<Scalar>) -> Self {
    let mut entries: Vec<(usize, Scalar)> = Vec::new();

    for (dense_index, dense_entry) in dense.Z.iter().enumerate() {
      if *dense_entry != Scalar::ZERO {
        entries.push((dense_index, *dense_entry));
      }
    }

    Self::new(dense.num_vars, entries)
  }
}

/// Number of lanes over which this SparseParPolynomial is chunked – ideal number 
/// ~= num_threads / (num polys in parallel)
/// Likely also want some slop for Rayon workstealing.
pub const SPARSE_CHUNKS: usize = 64;

/// TODO: I AM A FUCKING STRUCT.
#[derive(Debug, Clone)]
pub struct SparseParPolynomial<Scalar: PrimeField> {
  /// 
  pub num_vars: usize,
  ///
  pub Z: [Vec<(usize, Scalar)>; SPARSE_CHUNKS],
}

impl<Scalar: PrimeField> SparseParPolynomial<Scalar> {
  /// TODO:Documentation.
  pub fn from_non_par(non_par: SparsePolynomial<Scalar>) -> Self {
    let len = 1 << non_par.num_vars;
    let chunk_size = len / SPARSE_CHUNKS;

    // TODO(sragss): Interm solution to figure out what to do with chunking.
    let mut chunked_Z = core::array::from_fn(|_| Vec::new());
    for (sparse_index, value) in non_par.Z {
      let chunk = sparse_index / chunk_size;
      chunked_Z[chunk].push((sparse_index, value));
    }

    SparseParPolynomial { num_vars: non_par.num_vars, Z: chunked_Z }
  }

    /// TODO: Documentation
    #[tracing::instrument(skip_all, name = "SparsePolynomial::bound_poly_var_bot")]
    pub fn bound_poly_var_bot(&mut self, r: &Scalar) {
      self.Z.par_iter_mut().for_each(|Z| {
        let mut sparse_read_index = 0;
        let mut sparse_write_index = 0;
        while sparse_read_index < Z.len() {
          let a = Z[sparse_read_index];
    
          // Case where both low, high are non-sparse.
          if a.0 % 2 == 0
              && sparse_read_index != Z.len() - 1
              && a.0 == Z[sparse_read_index+1].0 - 1  {
            let b = Z[sparse_read_index+1];
    
            Z[sparse_write_index] = (a.0/ 2, a.1 + *r * (b.1 - a.1));
            sparse_read_index += 2;
            sparse_write_index += 1;
          } else {
    
            // if a.0 % 2 == 0 { // low
            //   Z[sparse_write_index] = (a.0 / 2, a.1 + *r * (-a.1));
            // } else { // high
            //   Z[sparse_write_index] = (a.0 / 2, *r * a.1);
            // }
            if a.0 % 2 == 0 { // low
              Z[sparse_write_index] = (a.0 / 2, (Scalar::ONE - *r) * a.1);
            } else { // high
              Z[sparse_write_index] = (a.0 / 2, *r * a.1);
            }
    
            sparse_read_index += 1;
            sparse_write_index += 1;
          }
    
        }
        Z.truncate(sparse_write_index);
      });
      self.num_vars -= 1;
    }
}

/// Adds another multilinear polynomial to `self`.
/// Assumes the two polynomials have the same number of variables.
impl<Scalar: PrimeField> Add for MultilinearPolynomial<Scalar> {
  type Output = Result<Self, &'static str>;

  fn add(self, other: Self) -> Self::Output {
    if self.get_num_vars() != other.get_num_vars() {
      return Err("The two polynomials must have the same number of variables");
    }

    let sum: Vec<Scalar> = self
      .Z
      .iter()
      .zip(other.Z.iter())
      .map(|(a, b)| *a + *b)
      .collect();

    Ok(MultilinearPolynomial::new(sum))
  }
}

#[cfg(test)]
mod tests {
  use crate::provider::{self, bn256_grumpkin::bn256, secp_secq::secp256k1};

  use super::*;
  use pasta_curves::Fp;

  fn make_mlp<F: PrimeField>(len: usize, value: F) -> MultilinearPolynomial<F> {
    MultilinearPolynomial {
      num_vars: len.count_ones() as usize,
      Z: vec![value; len],
    }
  }

  fn test_multilinear_polynomial_with<F: PrimeField>() {
    // Let the polynomial has 3 variables, p(x_1, x_2, x_3) = (x_1 + x_2) * x_3
    // Evaluations of the polynomial at boolean cube are [0, 0, 0, 1, 0, 1, 0, 2].

    let TWO = F::from(2);

    let Z = vec![
      F::ZERO,
      F::ZERO,
      F::ZERO,
      F::ONE,
      F::ZERO,
      F::ONE,
      F::ZERO,
      TWO,
    ];
    let m_poly = MultilinearPolynomial::<F>::new(Z.clone());
    assert_eq!(m_poly.get_num_vars(), 3);

    let x = vec![F::ONE, F::ONE, F::ONE];
    assert_eq!(m_poly.evaluate(x.as_slice()), TWO);

    let y = MultilinearPolynomial::<F>::evaluate_with(Z.as_slice(), x.as_slice());
    assert_eq!(y, TWO);
  }

  fn test_sparse_polynomial_with<F: PrimeField>() {
    // Let the polynomial have 3 variables, p(x_1, x_2, x_3) = (x_1 + x_2) * x_3
    // Evaluations of the polynomial at boolean cube are [0, 0, 0, 1, 0, 1, 0, 2].

    let TWO = F::from(2);
    let Z = vec![(3, F::ONE), (5, F::ONE), (7, TWO)];
    let m_poly = SparsePolynomial::<F>::new(3, Z);

    let x = vec![F::ONE, F::ONE, F::ONE];
    assert_eq!(m_poly.evaluate(x.as_slice()), TWO);

    let x = vec![F::ONE, F::ZERO, F::ONE];
    assert_eq!(m_poly.evaluate(x.as_slice()), F::ONE);
  }

  #[test]
  fn test_multilinear_polynomial() {
    test_multilinear_polynomial_with::<Fp>();
  }

  #[test]
  fn test_sparse_polynomial() {
    test_sparse_polynomial_with::<Fp>();
  }

  fn test_mlp_add_with<F: PrimeField>() {
    let mlp1 = make_mlp(4, F::from(3));
    let mlp2 = make_mlp(4, F::from(7));

    let mlp3 = mlp1.add(mlp2).unwrap();

    assert_eq!(mlp3.Z, vec![F::from(10); 4]);
  }

  fn test_mlp_scalar_mul_with<F: PrimeField>() {
    let mlp = make_mlp(4, F::from(3));

    let mlp2 = mlp.scalar_mul(&F::from(2));

    assert_eq!(mlp2.Z, vec![F::from(6); 4]);
  }

  #[test]
  fn test_mlp_add() {
    test_mlp_add_with::<Fp>();
    test_mlp_add_with::<bn256::Scalar>();
    test_mlp_add_with::<secp256k1::Scalar>();
  }

  #[test]
  fn test_mlp_scalar_mul() {
    test_mlp_scalar_mul_with::<Fp>();
    test_mlp_scalar_mul_with::<bn256::Scalar>();
    test_mlp_scalar_mul_with::<secp256k1::Scalar>();
  }

  fn test_evaluation_with<F: PrimeField>() {
    let num_evals = 4;
    let mut evals: Vec<F> = Vec::with_capacity(num_evals);
    for _ in 0..num_evals {
      evals.push(F::from_u128(8));
    }
    let dense_poly: MultilinearPolynomial<F> = MultilinearPolynomial::new(evals.clone());

    // Evaluate at 3:
    // (0, 0) = 1
    // (0, 1) = 1
    // (1, 0) = 1
    // (1, 1) = 1
    // g(x_0,x_1) => c_0*(1 - x_0)(1 - x_1) + c_1*(1-x_0)(x_1) + c_2*(x_0)(1-x_1) + c_3*(x_0)(x_1)
    // g(3, 4) = 8*(1 - 3)(1 - 4) + 8*(1-3)(4) + 8*(3)(1-4) + 8*(3)(4) = 48 + -64 + -72 + 96  = 8
    // g(5, 10) = 8*(1 - 5)(1 - 10) + 8*(1 - 5)(10) + 8*(5)(1-10) + 8*(5)(10) = 96 + -16 + -72 + 96  = 8
    assert_eq!(
      dense_poly.evaluate(vec![F::from(3), F::from(4)].as_slice()),
      F::from(8)
    );
  }

  #[test]
  fn test_evaluation() {
    test_evaluation_with::<Fp>();
    test_evaluation_with::<provider::bn256_grumpkin::bn256::Scalar>();
    test_evaluation_with::<provider::secp_secq::secp256k1::Scalar>();
  }

  use ff::Field;

  fn gen_dense_sparse_poly<F: PrimeField>(num_vars: usize, pct_sparse: f64) -> (MultilinearPolynomial<F>, SparsePolynomial<F>) {
    let num_entries: usize = 1 << num_vars;
    let mut dense_evals = Vec::new();
    let mut sparse_entries: Vec<(usize, F)> = Vec::new();
  
    for dense_index in 0..num_entries {
      let chance: f64 = rand::random();
      if chance > pct_sparse {
        let random_fp = F::random(&mut rand::thread_rng());
        dense_evals.push(random_fp);
        sparse_entries.push((dense_index, random_fp));
      } else {
        dense_evals.push(F::ZERO);
      }
    }
  
    (MultilinearPolynomial::new(dense_evals), SparsePolynomial::new(num_vars, sparse_entries))
  }

  #[test]
  fn test_sparse_parity() {
    let num_vars = 8;
    let pct_sparse: f64 = 0.6;

    let (dense, sparse) = gen_dense_sparse_poly::<Fp>(num_vars, pct_sparse);
    assert_eq!(dense, sparse.to_dense());
  }

  #[test]
  fn test_sparse_parity_bound_bot() {
    let num_vars = 8;
    let pct_sparse: f64 = 0.6;

    let (mut dense, mut sparse) = gen_dense_sparse_poly::<Fp>(num_vars, pct_sparse);

    let r = Fp::random(&mut rand::thread_rng());
    dense.bound_poly_var_bot(&r);
    sparse.bound_poly_var_bot(&r);
    assert_eq!(dense, sparse.to_dense());
  }
}
