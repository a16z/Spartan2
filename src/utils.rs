use ff::PrimeField;

/// a * b where there is a high P(a = {0, 1}) * P(b = {0, 1})
#[inline(always)]
pub fn mul_0_1_optimized<F: PrimeField>(a: &F, b: &F) -> F {
    if a.is_zero().into() || b.is_zero().into() {
        F::ZERO
    } else if *a == F::ONE {
        *b
    } else if *b == F::ONE {
        *a
    } else {
        *a * b
    }
}