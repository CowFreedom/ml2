extern crate rand;

use core::{Distribution, PDF,
ConditionalDistribution,ConditionalPDF};
use Float;
use std::cmp;
use self::rand::{FromEntropy, Rng};
use std;
use std::marker::PhantomData;

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
pub struct MetropolisHastings<D,F>
    where D:ConditionalDistribution+ConditionalPDF,
    D::T:AsRef<[Float]>+std::fmt::Debug+Clone,//+Copy wegen x0 inital move
    F:Fn(&[Float])->Float
    {
    currentSamples:usize,
    f:F,
    proposal:D,
    initial:D::T,
    markovIterations:usize,
    burnIn:usize,
    isSymmetric:bool,
    //_marker: PhantomData<&'a()>,
}

impl<D,F> MetropolisHastings<D,F>
    where D:ConditionalDistribution+ConditionalPDF,
    D::T:AsRef<[Float]>+std::fmt::Debug+Clone,
    F:Fn(&[Float])->Float
    {

    pub fn new(f:F,proposal:D,x0:D::T)->MetropolisHastings<D,F>
    {
        MetropolisHastings{
        f,
        proposal,
        initial:x0,
        markovIterations:1000,
        burnIn:(250 as Float) as usize,
        isSymmetric:false,
        currentSamples:0,
       //_marker: PhantomData::<&'a()>,
        }

    }

    //create initial values for the start of the Markov Chain
    fn createInitialValues(&self)->Vec<Float>{
        let x=vec![0.0;1];
        x
    }

    pub fn sample(&self, n:usize)->Vec<Vec<Float>>{
        let mut samples=Vec::with_capacity(n);

        //let mut x=self.createInitialValues();
        let mut x=self.initial.clone();

        for i in 0..self.burnIn+n as usize{
            let x_p=self.proposal.csample(&mut rand::thread_rng(),&x);
            let accRat=(self.f)(x_p.as_ref())*self.proposal.cpdf(&x_p,&x)/((self.f)(x.as_ref())*self.proposal.cpdf(&x,&x_p));
            //println!("Proposal x_P: {:?} Proposal x: {:?}",self.proposal.pdf(&x_p),self.proposal.pdf(&x));
            //let accRat=((self.f)(x_p.as_ref()))/((self.f)(x.as_ref()));
            let r=f64::min(1.0,accRat);//change precision
            let mut rng=rand::thread_rng();
            if rng.gen::<Float>() <r{
                x.copy_from_slice(x_p.as_ref()); //x=x_p
            }
            if i>=self.burnIn{
                let mut k=vec![0.0];
                k.copy_from_slice(x.as_ref());

                samples.push(k);
            } 
        }

        //println!("Samples: {:?}",samples);
        samples

    }



}


