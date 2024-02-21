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

        let map_field_elements_to_u64 = |field_elements: &[Self::Scalar]| {
          field_elements
            .iter()
            .map(|field_element| {
              let limbs: [u64; 4] = (*field_element).into();
              limbs[0]
            })
            .collect()
        };

        match max_num_bits {
          0 => Self::zero(),
          1 => {
            let scalars_u64: Vec<_> = map_field_elements_to_u64(scalars);
            Self::msm_binary(bases, &scalars_u64)
          }
          2..=10 => {
            let scalars_u64: Vec<_> = map_field_elements_to_u64(scalars);
            Self::msm_small(bases, &scalars_u64, max_num_bits)
          }
          11..=64 => {
            let scalars_u64: Vec<_> = map_field_elements_to_u64(scalars);
            Self::msm_u64_wnaf(bases, &scalars_u64, max_num_bits)
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

      // Adapted from the Jolt implementation, which is in turn adapted from the ark_ec implementation
      fn msm_u64_wnaf(
        bases: &[Self::PreprocessedGroupElement],
        scalars: &[u64],
        max_num_bits: usize,
      ) -> Self {
        use std::cmp::Ordering;

        let c = if bases.len() < 32 {
          3
        } else {
          // ~= ln(bases.len()) + 2
          (bases.len().ilog2() as usize * 69 / 100) + 2
        };

        let digits_count = (max_num_bits + c - 1) / c;
        let radix: u64 = 1 << c;
        let window_mask: u64 = radix - 1;

        let scalar_digits = scalars
          .into_par_iter()
          .flat_map_iter(|&scalar| {
            let mut carry = 0u64;
            (0..digits_count).into_iter().map(move |i| {
              // Construct a buffer of bits of the scalar, starting at `bit_offset`.
              let bit_offset = i * c;
              let bit_idx = bit_offset % 64;
              // Read the bits from the scalar
              let bit_buf = scalar >> bit_idx;
              // Read the actual coefficient value from the window
              let coef = carry + (bit_buf & window_mask); // coef = [0, 2^r)

              // Recenter coefficients from [0,2^c) to [-2^c/2, 2^c/2)
              carry = (coef + radix / 2) >> c;
              let mut digit = (coef as i64) - (carry << c) as i64;

              if i == digits_count - 1 {
                digit += (carry << c) as i64;
              }
              digit
            })
          })
          .collect::<Vec<_>>();
        let zero = Self::zero();

        let window_sums: Vec<_> = (0..digits_count)
          .into_par_iter()
          .map(|i| {
            let mut buckets = vec![zero; 1 << c];
            for (digits, base) in scalar_digits.chunks(digits_count).zip(bases) {
              // digits is the digits thing of the first scalar?
              let scalar = digits[i];
              match 0.cmp(&scalar) {
                Ordering::Less => buckets[(scalar - 1) as usize] += base,
                Ordering::Greater => buckets[(-scalar - 1) as usize] -= base,
                Ordering::Equal => (),
              }
            }

            let mut running_sum = Self::zero();
            let mut res = Self::zero();
            buckets.iter().rev().for_each(|b| {
              running_sum += b;
              res += &running_sum;
            });
            res
          })
          .collect();

        // We store the sum for the lowest window.
        let lowest = *window_sums.first().unwrap();

        // We're traversing windows from high to low.
        let result = lowest
          + window_sums[1..]
            .iter()
            .rev()
            .fold(zero, |mut total, sum_i| {
              total += sum_i;
              for _ in 0..c {
                total = total.double();
              }
              total
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
