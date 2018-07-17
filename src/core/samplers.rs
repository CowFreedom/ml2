extern crate rand;

use core::{Distribution, PDF};
use Float;
use std::cmp;
use self::rand::{FromEntropy, Rng};
use std;


#[inline]
pub fn RejectionSampleRegion<D,F>(dist:&D, f:F)-> D::T
    where D:Distribution,
    F:Fn(&D::T)->bool
    {
      let mut x=dist.sample(&mut rand::thread_rng());
        while !f(&x){
            x=dist.sample(&mut rand::thread_rng())
        }
        x
}

//pub fn MCMC
// richtige Dr Arbeit
pub struct MetropolisHastings<D,F>
    where D:Distribution+PDF,
    D::T:AsRef<[Float]>+std::fmt::Debug,
    F:Fn(&[Float])->Float
    {
    currentSamples:usize,
    f:F,
    proposal:D,
    acceptance:D,
    markovIterations:usize,
    burnIn:usize,
    isSymmetric:bool
}

impl<D,F> MetropolisHastings<D,F>
    where D:Distribution+PDF,
    D::T:AsRef<[Float]>+std::fmt::Debug,
    F:Fn(&[Float])->Float
    {

    pub fn new(f:F,proposal:D,acceptance:D)->MetropolisHastings<D,F>
    {
        MetropolisHastings{
        f,
        acceptance,
        proposal,
        markovIterations:1000,
        burnIn:(250 as Float) as usize,
        isSymmetric:false,
        currentSamples:0
        }

    }

    pub fn sample(&self, n:usize){
        let mut samples=Vec::with_capacity(n);

        let mut x=self.proposal.sample(&mut rand::thread_rng());

        for i in 0..self.burnIn+n as usize{
            let x_p=self.proposal.sample(&mut rand::thread_rng());
            let accRat=((self.f)(x_p.as_ref())*self.proposal.pdf(&x_p))/((self.f)(x.as_ref())*self.proposal.pdf(&x));
            let r=f64::min(1.0,accRat);//change precision
            let mut rng=rand::thread_rng();
            if rng.gen::<Float>() <r{
                x=x_p;
            }
            if i>=self.burnIn{
                let mut k=[0.0];
                println!("clone:{:?}",k.copy_from_slice(x.as_ref()));
                samples.push(k.clone_from_slice(x.as_ref()));
            } 
        }

        println!("Samples: {:?}",samples);


    }



}


