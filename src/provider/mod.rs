//! This module implements Spartan's traits using the following configuration:
//! `CommitmentEngine` with Pedersen's commitments
//! `Group` with pasta curves and BN256/Grumpkin
//! `RO` traits with Poseidon
//! `EvaluationEngine` with an IPA-based polynomial evaluation argument

pub mod bn256_grumpkin;
pub mod hyrax_pc;
pub mod ipa_pc;
pub mod keccak;
pub mod pasta;
pub mod pedersen;
pub mod secp_secq;

/// Curve ops
/// This implementation behaves in ways specific to the halo2curves suite of curves in:
// - to_coordinates,
// - vartime_multiscalar_mul, where it does not call into accelerated implementations.
// A specific reimplementation exists for the pasta curves in their own module.
#[macro_export]
macro_rules! impl_traits {
  (
    $name:ident,
    $name_compressed:ident,
    $name_curve:ident,
    $name_curve_affine:ident,
    $order_str:literal
  ) => {
    impl Group for $name::Point {
      type Base = $name::Base;
      type Scalar = $name::Scalar;
      type CompressedGroupElement = $name_compressed;
      type PreprocessedGroupElement = $name::Affine;
      type TE = Keccak256Transcript<Self>;
      type CE = HyraxCommitmentEngine<Self>;

      fn vartime_multiscalar_mul(
        scalars: &[Self::Scalar],
        bases: &[Self::PreprocessedGroupElement],
      ) -> Self {
        let max_num_bits = scalars
          .par_iter()
          .map(|s| {
            let le_bits = s.to_le_bits();
            le_bits.len() - le_bits.trailing_zeros()
          })
          .max()
          .unwrap();

        match max_num_bits {
          0 => Self::zero(),
          1 => {
            let scalars_u64: Vec<_> = scalars
              .iter()
              .map(|scalar| {
                let limbs: [u64; 4] = (*scalar).into();
                limbs[0]
              })
              .collect();
            Self::msm_binary(bases, &scalars_u64)
          }
          2..=10 => {
            let scalars_u64: Vec<_> = scalars
              .iter()
              .map(|scalar| {
                let limbs: [u64; 4] = (*scalar).into();
                limbs[0]
              })
              .collect();
            Self::msm_small(bases, &scalars_u64, max_num_bits)
          }
          _ => best_multiexp(scalars, bases),
        }
      }

      fn msm_binary(bases: &[Self::PreprocessedGroupElement], scalars: &[u64]) -> Self {
        scalars
          .iter()
          .zip(bases)
          .filter(|(&scalar, _base)| scalar != 0)
          .map(|(_scalar, base)| base)
          .fold(Self::zero(), |sum, base| sum + base)
      }

      fn msm_small(
        bases: &[Self::PreprocessedGroupElement],
        scalars: &[u64],
        max_num_bits: usize,
      ) -> Self {
        let num_buckets: usize = 1 << max_num_bits;
        // Assign things to buckets based on the scalar
        let mut buckets: Vec<Self> = vec![Self::zero(); num_buckets];
        scalars
          .iter()
          .zip(bases)
          .filter(|(&scalar, _base)| scalar != 0)
          .for_each(|(&scalar, base)| {
            buckets[scalar as usize] += base;
          });

        let mut result = Self::zero();
        let mut running_sum = Self::zero();
        buckets.iter().skip(1).rev().for_each(|bucket| {
          running_sum += bucket;
          result += running_sum;
        });
        result
      }

      fn preprocessed(&self) -> Self::PreprocessedGroupElement {
        self.to_affine()
      }

      fn compress(&self) -> Self::CompressedGroupElement {
        self.to_bytes()
      }

      fn from_label(label: &'static [u8], n: usize) -> Vec<Self::PreprocessedGroupElement> {
        let mut shake = Shake256::default();
        shake.update(label);
        let mut reader = shake.finalize_xof();
        let mut uniform_bytes_vec = Vec::new();
        for _ in 0..n {
          let mut uniform_bytes = [0u8; 32];
          reader.read_exact(&mut uniform_bytes).unwrap();
          uniform_bytes_vec.push(uniform_bytes);
        }
        let gens_proj: Vec<$name_curve> = (0..n)
          .into_par_iter()
          .map(|i| {
            let hash = $name_curve::hash_to_curve("from_uniform_bytes");
            hash(&uniform_bytes_vec[i])
          })
          .collect();

        let num_threads = rayon::current_num_threads();
        if gens_proj.len() > num_threads {
          let chunk = (gens_proj.len() as f64 / num_threads as f64).ceil() as usize;
          (0..num_threads)
            .into_par_iter()
            .flat_map(|i| {
              let start = i * chunk;
              let end = if i == num_threads - 1 {
                gens_proj.len()
              } else {
                core::cmp::min((i + 1) * chunk, gens_proj.len())
              };
              if end > start {
                let mut gens = vec![$name_curve_affine::identity(); end - start];
                <Self as Curve>::batch_normalize(&gens_proj[start..end], &mut gens);
                gens
              } else {
                vec![]
              }
            })
            .collect()
        } else {
          let mut gens = vec![$name_curve_affine::identity(); n];
          <Self as Curve>::batch_normalize(&gens_proj, &mut gens);
          gens
        }
      }

      fn to_coordinates(&self) -> (Self::Base, Self::Base, bool) {
        // see: grumpkin implementation at src/provider/bn256_grumpkin.rs
        let coordinates = self.to_affine().coordinates();
        if coordinates.is_some().unwrap_u8() == 1
          && (Self::PreprocessedGroupElement::identity() != self.to_affine())
        {
          (*coordinates.unwrap().x(), *coordinates.unwrap().y(), false)
        } else {
          (Self::Base::zero(), Self::Base::zero(), true)
        }
      }

      fn get_curve_params() -> (Self::Base, Self::Base, BigInt) {
        let A = $name::Point::a();
        let B = $name::Point::b();
        let order = BigInt::from_str_radix($order_str, 16).unwrap();

        (A, B, order)
      }

      fn zero() -> Self {
        $name::Point::identity()
      }

      fn get_generator() -> Self {
        $name::Point::generator()
      }
    }

    impl PrimeFieldExt for $name::Scalar {
      fn from_uniform(bytes: &[u8]) -> Self {
        let bytes_arr: [u8; 64] = bytes.try_into().unwrap();
        $name::Scalar::from_uniform_bytes(&bytes_arr)
      }
    }

    impl<G: Group> TranscriptReprTrait<G> for $name_compressed {
      fn to_transcript_bytes(&self) -> Vec<u8> {
        self.as_ref().to_vec()
      }
    }

    impl CompressedGroup for $name_compressed {
      type GroupElement = $name::Point;

      fn decompress(&self) -> Option<$name::Point> {
        Some($name_curve::from_bytes(&self).unwrap())
      }
    }
  };
}
