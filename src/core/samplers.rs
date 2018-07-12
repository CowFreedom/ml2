extern crate rand;

use core::Distribution;
use Float;

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


