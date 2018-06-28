extern crate rand;

use core::Distribution;
use Float;

#[inline]
pub fn RejectionSampleRegion<D>(dist:&D, f:&Fn(&D::T)->bool)-> D::T
    where D:Distribution
    {
      let mut x=dist.sample(&mut rand::thread_rng());
        while !f(&x){
            x=dist.sample(&mut rand::thread_rng())
        }
        x
}


