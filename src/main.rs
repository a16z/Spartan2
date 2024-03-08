use spartan2::{provider::bn256_grumpkin::bn256::Scalar as BnFp, spartan::polys::multilinear::SparseParPolynomial};
use ff::Field;
use clap::{Parser, CommandFactory};
use rayon::prelude::*;

/// Configuration for the application
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Flag to run sparse benchmark in parallel
    #[arg(short, long)]
    sparse_poly: bool,

    /// Flag to run circuits
    #[arg(short, long)]
    circuits: bool,
}

fn main() {
    let args = Args::parse();

    let (chrome_layer, _guard) = ChromeLayerBuilder::new().build();
    tracing_subscriber::registry().with(chrome_layer).init();

    if args.sparse_poly {
        println!("Sparse polynomial binding bench 2^20 evals, 70% sparse, 2 polys in parallel");
        run_sparse_bench_parallel(20, 0.7, 2);

        println!("\n\nSparse polynomial binding bench 2^20 evals, 70% sparse, 32 polys in parallel");
        run_sparse_bench_parallel(20, 0.7, 32);

        println!("\n\nSparse polynomial binding bench 2^26 evals, 70% sparse, 2 polys in parallel");
        run_sparse_bench_parallel(26, 0.7, 2);

        println!("\n\nSparse polynomial binding bench 2^26 evals, 70% sparse, 4 polys in parallel");
        run_sparse_bench_parallel(26, 0.7, 4);

        println!("\n\nSparse polynomial binding bench 2^26 evals, 90% sparse, 2 polys in parallel");
        run_sparse_bench_parallel(26, 0.9, 2);

        println!("\n\nSparse polynomial binding bench 2^26 evals, 80% sparse, 2 polys in parallel");
        run_sparse_bench_parallel(26, 0.8, 2);

        println!("\n\nSparse polynomial binding bench 2^26 evals, 60% sparse, 2 polys in parallel");
        run_sparse_bench_parallel(26, 0.6, 2);

        println!("\n\nSparse polynomial binding bench 2^26 evals, 50% sparse, 2 polys in parallel");
        run_sparse_bench_parallel(26, 0.5, 2);

        println!("\n\nSparse polynomial binding bench 2^26 evals, 40% sparse, 2 polys in parallel");
        run_sparse_bench_parallel(26, 0.4, 2);

        println!("\n\nSparse polynomial binding bench 2^26 evals, 30% sparse, 2 polys in parallel");
        run_sparse_bench_parallel(26, 0.3, 2);
    } else if args.circuits {
        run_circuits();
    } else {
        Args::command().print_help().expect("Failed to print help information");
        std::process::exit(1);
    }

    drop(_guard);
}

fn run_circuits() {
    let circuits = vec![
        Sha256Circuit::new(vec![0u8; 1 << 12]),
    ];

    for circuit in circuits {
        let span = tracing::info_span!("SpartanProve-Sha256-message-len", len = circuit.preimage.len());
        let _enter = span.enter();

        run_circuit(circuit);
    }
}

fn run_circuit(circuit: Sha256Circuit<<G as Group>::Scalar>) {
    let start = std::time::Instant::now();

    // produce keys
    let (pk, _vk) =
        SNARK::<G, S, Sha256Circuit<<G as Group>::Scalar>>::setup(circuit.clone()).unwrap();

    let _ = SNARK::<G, S, Sha256Circuit<<G as Group>::Scalar>>::prove(
        &pk,
        circuit.clone(),
    );

    let duration = start.elapsed();
    println!("Time elapsed is: {:?}", duration);
}

fn gen_dense_sparse_poly(num_vars: usize, pct_sparse: f64) -> (MultilinearPolynomial<BnFp>, SparsePolynomial<BnFp>) {
  let num_entries: usize = 1 << num_vars;
  let mut dense_evals = Vec::new();
  let mut sparse_entries: Vec<(usize, BnFp)> = Vec::new();

  for dense_index in 0..num_entries {
    let chance: f64 = rand::random();
    if chance > pct_sparse {
      let random_fp = BnFp::random(&mut rand::thread_rng());
      dense_evals.push(random_fp);
      sparse_entries.push((dense_index, random_fp));
    } else {
      dense_evals.push(BnFp::zero());
    }
  }

  (MultilinearPolynomial::new(dense_evals), SparsePolynomial::new(num_vars, sparse_entries))
}


fn run_sparse_bench_parallel(num_vars: usize, pct_sparse: f64, parallelism: usize) {
  let mut dense_polys: Vec<MultilinearPolynomial<BnFp>> = Vec::new();
  let mut sparse_polys: Vec<SparsePolynomial<BnFp>> = Vec::new();

  for _ in 0..parallelism {
    let (dense, sparse) = gen_dense_sparse_poly(num_vars, pct_sparse);
    dense_polys.push(dense);
    sparse_polys.push(sparse);
  }

  let r = BnFp::random(&mut rand::thread_rng());

  let mut dense_top = dense_polys.clone();
  let dense_top_start = std::time::Instant::now();
  dense_top.par_iter_mut().for_each(|dense_poly| dense_poly.bound_poly_var_top(&r));
  let dense_top_duration = dense_top_start.elapsed();
  println!("Time elapsed for bounding dense polynomials (top, par):                {:?}", dense_top_duration);

  let mut dense_top= dense_polys.clone();
  let dense_top_start = std::time::Instant::now();
  dense_top.par_iter_mut().for_each(|dense_poly| dense_poly.bound_poly_var_top_zero_optimized(&r));
  let dense_top_duration = dense_top_start.elapsed();
  println!("Time elapsed for bounding dense polynomials (top, par) zero optimized: {:?}", dense_top_duration);

  let mut dense_regular = dense_polys.clone();
  let dense_start = std::time::Instant::now();
  dense_regular.par_iter_mut().for_each(|dense_poly| dense_poly.bound_poly_var_bot(&r));
  let dense_duration = dense_start.elapsed();
  println!("Time elapsed for bounding dense polynomials (bot):                     {:?}", dense_duration);

  let mut dense_zero_optimized = dense_polys.clone();
  let dense_start = std::time::Instant::now();
  dense_zero_optimized.par_iter_mut().for_each(|dense_poly| dense_poly.bound_poly_var_bot_zero_optimized(&r));
  let dense_duration = dense_start.elapsed();
  println!("Time elapsed for bounding dense polynomials zero optimized:            {:?}", dense_duration);

  let mut sparse_regular = sparse_polys.clone();
  let sparse_start = std::time::Instant::now();
  sparse_regular.par_iter_mut().for_each(|sparse_poly| sparse_poly.bound_poly_var_bot(&r));
  let sparse_duration = sparse_start.elapsed();
  println!("Time elapsed for bounding sparse polynomials:                          {:?}", sparse_duration);

  let mut sparse_par: Vec<SparseParPolynomial<BnFp>> = sparse_polys.clone().into_par_iter().map(|sparse_poly| SparseParPolynomial::from_non_par(sparse_poly)).collect();;
  let sparse_par_start = std::time::Instant::now();
  sparse_par.par_iter_mut().for_each(|sparse_poly| sparse_poly.bound_poly_var_bot(&r));
  let sparse_duration = sparse_par_start.elapsed();
  println!("Time elapsed for bounding sparse par polynomials:                      {:?}", sparse_duration);

  for i in 0..parallelism {
    // assert_eq!(dense_regular[i], dense_zero_optimized[i]);
    assert_eq!(dense_regular[i], sparse_regular[i].clone().to_dense());
  }




  // Items to attempt
  // - Regular bound_poly_var_bot
  // - 0 / 1 optimized
  // - High parallelism bound_poly_var_bot
  // 
}

use bellpepper::gadgets::{sha256::sha256, Assignment};
use bellpepper_core::{
  boolean::{AllocatedBit, Boolean},
  num::{AllocatedNum, Num},
  Circuit, ConstraintSystem, SynthesisError,
};
use ff::{PrimeField, PrimeFieldBits};
use sha2::{Digest, Sha256};
use spartan2::{spartan::polys::multilinear::{MultilinearPolynomial, SparsePolynomial}, traits::Group, SNARK};
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use std::marker::PhantomData;

type G = pasta_curves::pallas::Point;
type EE = spartan2::provider::hyrax_pc::HyraxEvaluationEngine<G>;
type S = spartan2::spartan::upsnark::R1CSSNARK<G, EE>;

#[derive(Clone, Debug)]
struct Sha256Circuit<Scalar: PrimeField> {
  preimage: Vec<u8>,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField + PrimeFieldBits> Sha256Circuit<Scalar> {
  pub fn new(preimage: Vec<u8>) -> Self {
    Self {
      preimage,
      _p: PhantomData,
    }
  }
}

impl<Scalar: PrimeField> Circuit<Scalar> for Sha256Circuit<Scalar> {
  fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let bit_values: Vec<_> = self
      .preimage
      .clone()
      .into_iter()
      .flat_map(|byte| (0..8).map(move |i| (byte >> i) & 1u8 == 1u8))
      .map(Some)
      .collect();
    assert_eq!(bit_values.len(), self.preimage.len() * 8);

    let preimage_bits = bit_values
      .into_iter()
      .enumerate()
      .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {i}")), b))
      .map(|b| b.map(Boolean::from))
      .collect::<Result<Vec<_>, _>>()?;

    let hash_bits = sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

    for (i, hash_bits) in hash_bits.chunks(256_usize).enumerate() {
      let mut num = Num::<Scalar>::zero();
      let mut coeff = Scalar::ONE;
      for bit in hash_bits {
        num = num.add_bool_with_coeff(CS::one(), bit, coeff);

        coeff = coeff.double();
      }

      let hash = AllocatedNum::alloc(cs.namespace(|| format!("input {i}")), || {
        Ok(*num.get_value().get()?)
      })?;

      // num * 1 = hash
      cs.enforce(
        || format!("packing constraint {i}"),
        |_| num.lc(Scalar::ONE),
        |lc| lc + CS::one(),
        |lc| lc + hash.get_variable(),
      );
    }

    // sanity check with the hasher
    let mut hasher = Sha256::new();
    hasher.update(&self.preimage);
    let hash_result = hasher.finalize();

    let mut s = hash_result
      .iter()
      .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1u8 == 1u8));

    for b in hash_bits {
      match b {
        Boolean::Is(b) => {
          assert!(s.next().unwrap() == b.get_value().unwrap());
        }
        Boolean::Not(b) => {
          assert!(s.next().unwrap() != b.get_value().unwrap());
        }
        Boolean::Constant(_b) => {
          panic!("Can't reach here")
        }
      }
    }
    Ok(())
  }
}
