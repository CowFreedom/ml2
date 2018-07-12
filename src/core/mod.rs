pub mod stat;
pub mod samplers;
pub mod integrators;


extern crate rand;
use ::Float;
use self::rand::Rng;

/*Standard trait for distributions. Some applications only
require the ability to sample from a distribution without the need
for an explicit probability density function (pdf).*/
pub trait Distribution
    {
    type T;
    fn sample<R:Rng+?Sized>(&self, rng:&mut R)->Self::T;
}

/*Standard trait for distributions. Some applications only
require the ability to sample from a distribution without the need
for an explicit probability density function (pdf).*/
pub trait PDF: Distribution
    {
    fn pdf(&self,x:&Self::T)->Float;
}