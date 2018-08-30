extern crate rand;

use core::{Distribution, PDF,
ConditionalDistribution,ConditionalPDF};
use Float;
use self::rand::{FromEntropy, Rng};
use std;
use std::iter::ExactSizeIterator;
extern crate num_cpus;
extern crate crossbeam;

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
    D::T:AsRef<[Float]>+std::fmt::Debug+Clone,
    F:Fn(&[Float])->Float
    {
    f:F,
    proposal:D,
    initial:D::T,
    burnIn:usize,
    isSymmetric:bool,
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
        burnIn:(250 as Float) as usize,
        isSymmetric:false,
        }

    }

        pub fn sample(&self, n:usize)->Vec<Float>{
            let dim=1;//ändern!
            let mut samples=Vec::with_capacity(n*dim);
         /*   
            let mut thread_handles=Vec::with_capacity(self.nThreads);

            crossbeam::scope(|scope|{
                let sample_chunks=sample.chunks_mut(n*dim);


            }
*/
            let mut rng=rand::thread_rng();
            let mut x=self.initial.clone();

            for i in 0..self.burnIn+n as usize{
                let x_p=self.proposal.csample(&mut rng,&x);
                let accRat=((self.f)(x_p.as_ref())*self.proposal.cpdf(&x_p,&x))/((self.f)(x.as_ref())*self.proposal.cpdf(&x,&x_p));
                let r=f64::min(1.0,accRat);
                if rng.gen::<Float>() <r{
                    x=x_p.clone();
                }
                if i>=self.burnIn{
                    samples.extend_from_slice(x_p.as_ref());
                } 
            }
            

          //  println!("Samples: {:?}",samples);
            samples

    }


}



//OLD MCMC
pub struct MetropolisHastings1D<D,F>
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
    nThreads:usize,
    //_marker: PhantomData<&'a()>,
}

impl<D,F> MetropolisHastings1D<D,F>
    where D:ConditionalDistribution+ConditionalPDF,
    D::T:AsRef<[Float]>+std::fmt::Debug+Clone,
    F:Fn(&[Float])->Float
    {

    pub fn new(f:F,proposal:D,x0:D::T)->MetropolisHastings1D<D,F>
    {
        MetropolisHastings1D{
        f,
        proposal,
        initial:x0,
        markovIterations:1000,
        burnIn:(250 as Float) as usize,
        isSymmetric:false,
        currentSamples:0,
        nThreads: num_cpus::get(),
       //_marker: PhantomData::<&'a()>,
        }

    }


    pub fn sample(&self, n:usize)->Vec<D::T>{
        let mut samples=Vec::with_capacity(n);
        let mut x=self.initial.clone();

        for i in 0..self.burnIn+n as usize{
            let x_p=self.proposal.csample(&mut rand::thread_rng(),&x);
            let accRat=((self.f)(x_p.as_ref())*self.proposal.cpdf(&x_p,&x))/((self.f)(x.as_ref())*self.proposal.cpdf(&x,&x_p));
            let r=f64::min(1.0,accRat);//change precision
            let mut rng=rand::thread_rng();
            if rng.gen::<Float>() <r{
                x=x_p.clone();
            }
            if i>=self.burnIn{
                samples.push(x.clone());
            } 
        }
        //println!("Samples: {:?}",samples);
        samples

    }



}


