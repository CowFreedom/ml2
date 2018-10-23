extern crate rand;

use core::{Distribution, PDF, ConditionalDistribution, ConditionalPDF};
use Float;
use self::rand::{FromEntropy, Rng};
use std;
use std::iter::ExactSizeIterator;
extern crate num_cpus;
extern crate crossbeam;

#[inline]
pub fn RejectionSampleRegion<D, F>(dist: &D, f: F) -> D::T
where
    D: Distribution,
    F: Fn(&D::T) -> bool,
{
    let mut x = dist.sample(&mut rand::thread_rng());
    while !f(&x) {
        x = dist.sample(&mut rand::thread_rng())
    }
    x
}

#[doc = "Markov Chain Metropolis Hastings algorithm. Numerically evaluate the value of an integral, where the integrand is proportional to
a probability distribution."]
pub struct MetropolisHastings<D, F>
where
    D: ConditionalDistribution + ConditionalPDF,
    D::T: AsRef<[Float]> + std::fmt::Debug + Clone,
    F: Fn(&[Float]) -> Float,
{
    log_f: F,
    log_proposal: D,
    initial: D::T,
    burnIn: usize,
    isSymmetric: bool,
}

impl<D, F> MetropolisHastings<D, F>
where
    D: ConditionalDistribution + ConditionalPDF,
    D::T: AsRef<[Float]> + std::fmt::Debug + Clone,
    F: Fn(&[Float]) -> Float,
{
    pub fn new(log_f: F, log_proposal: D, x0: D::T) -> MetropolisHastings<D, F> {
        MetropolisHastings {
            log_f,
            log_proposal,
            initial: x0,
            burnIn: (250 as Float) as usize,
            isSymmetric: false,
        }

    }

    pub fn sample(&self, n: usize) -> Vec<Float> {
        let dim = 1; //ändern!
        let mut samples = Vec::with_capacity(n * dim);
        let mut rng = rand::thread_rng();
        let mut x = self.initial.clone();

        for i in 0..self.burnIn + n as usize {
            let x_p = self.log_proposal.csample(&mut rng, &x);
            let accRat =(self.log_f)(x_p.as_ref()) + self.log_proposal.cpdf(&x_p, &x)
                       -(self.log_f)(x.as_ref()) -self.log_proposal.cpdf(&x, &x_p);
            let r = f64::min(0.0, accRat);
            let u = rng.gen::<Float>();
            if -0.7 * 2.0 * (u - 0.5) - 2.0 * (u - 0.5) * (u - 0.5) < r {
                //the polynomial is a truncated Taylor series of u.ln(x) at x=0.5
                x = x_p.clone();
            }
            if i >= self.burnIn {
                samples.extend_from_slice(x_p.as_ref());
            }
        }
        //  println!("Samples: {:?}",samples);
        samples
    }
}

pub struct gibbs<D, F>
where
    D: ConditionalDistribution + ConditionalPDF,
    D::T: AsRef<[Float]> + std::fmt::Debug + Clone,
    F: Fn(&[Float]) -> Float,
{
    log_f: F,
    log_proposal: D,
    initial: D::T,
    burnIn: usize,
    isSymmetric: bool,
}

pub fn gelman_convergence() {
    /*   
            let mut thread_handles=Vec::with_capacity(self.nThreads);

            crossbeam::scope(|scope|{
                let sample_chunks=sample.chunks_mut(n*dim);


            }
*/
}
