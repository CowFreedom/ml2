pub mod stat;
pub mod samplers;
pub mod integrators;


extern crate rand;
use Float;
use self::rand::Rng;

/*Standard trait for distributions. Some applications only
require the ability to sample from a distribution without the need
for an explicit probability density function (pdf).*/
pub trait Distribution {
    type T;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::T;
}

/*Standard trait for distributions. Some applications only
require the ability to sample from a distribution without the need
for an explicit probability density function (pdf).*/
pub trait PDF: Distribution {
    fn pdf(&self, x: &Self::T) -> Float;
}

/*When Rust offers optional function arguments, the conditional 
traits can be depraciated.*/
pub trait ConditionalDistribution {
    type T;
    fn csample<R: Rng + ?Sized>(&self, rng: &mut R, x_prev: &Self::T) -> Self::T;
}

/*Conditional probability density function. This is especially useful, if the
expression for the conditional pdf does only depend on the shape.*/
pub trait ConditionalPDF: ConditionalDistribution {
    fn cpdf(&self, x: &Self::T, x_prev: &Self::T) -> Float;
}
