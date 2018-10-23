extern crate rand;
use self::rand::{FromEntropy, Rng};
use self::rand::distributions::{Uniform, StandardNormal};
use self::rand::rngs::SmallRng;
use Float;
use core::{Distribution, PDF, ConditionalDistribution, ConditionalPDF};
use core;
use std::f64::consts;

use std::time::{SystemTime, UNIX_EPOCH};
use std;



//#[cfg(test)]

pub fn ProfileImportanceSampling() {
    enum Distributions {
        Normal,
        Uniform,
    }
    //HIER SIND PROBLEME
    impl Distribution for Distributions {
        type T = Float;
        fn sample<'a, R: Rng + ?Sized>(&'a self, rng: &mut R) -> Self::T {
            match self {
                Distributions::Normal => SmallRng::from_entropy().sample(StandardNormal),
                Distributions::Uniform => 4.0 * rng.gen::<f64>() - 2 as Float,
            }
        }
    }
    impl PDF for Distributions {
        fn pdf(&self, x: &Self::T) -> Float {
            match self {
                Distributions::Normal => 0.3989 * f64::powf(consts::E, -0.5 * x * x),
                Distributions::Uniform => if *x >= -2.0 && *x <= 2.0 { 0.25 } else { 0.0 },
            }
        }
    }
    let q = [Distributions::Normal, Distributions::Uniform];

    //Number of samples per sample distribution
    let k = 1_00_000_000;
    let n = [k, k];

    //Function to evaluate: s: sample, d:destination
    let f = |s: Float| -> Float {
        //f64::powf(consts::E,-1.0*s*s)
        if s < 2.0 && s > -2.0 {
            f64::powf(consts::E, s * s * s * s)
        } else {
            0.0
        }
    };

    let start1 = std::time::Instant::now();
    let res1 = core::integrators::MultipleImportanceSamplingSingleCore(&q, &n, &f);
    let timeSingleCore = start1.elapsed().as_secs();
    let start2 = std::time::Instant::now();
    let res2 = core::integrators::MultipleImportanceSampling(&q, &n, &f);
    let timeMultiCore = start2.elapsed().as_secs();
    println!(
        "SingleCore: {:?}\nMultiCore: {:?}",
        timeSingleCore,
        timeMultiCore
    );
    println!("res1:{}, res2:{}", res1, res2);



}


pub fn ProfileMetropolisSampling() {

    struct ConditionalDistributions {}


    impl ConditionalDistribution for ConditionalDistributions {
        type T = [Float; 1];
        #[inline]
        fn csample<R: Rng + ?Sized>(&self, rng: &mut R, x_prev: &Self::T) -> Self::T {
            [2.0 * rng.sample(StandardNormal) + x_prev[0]]
        }
    }
    impl ConditionalPDF for ConditionalDistributions {
        #[inline]
        fn cpdf(&self, x: &Self::T, x_prev: &Self::T) -> Float {
            //f64::exp(-0.5*0.25*(x[0]-x_prev[0])*(x[0]-x_prev[0]))
            -0.5 * 0.25 * (x[0] - x_prev[0]) * (x[0] - x_prev[0])
        }
    }

    //Function to evaluate which is proportional to a valid pdf
    let f = |s: &[Float]| -> Float {
        //f64::exp(-0.5*(s[0]-2.0)*(s[0]-2.0))
        -0.5 * (s[0] - 2.0) * (s[0] - 2.0)
    };

    let proposal = ConditionalDistributions {};

    let mut x0 = [0.0];

    let x = core::samplers::MetropolisHastings::new(f, proposal, x0);
    let n = 2_000_000_000;
    let start1 = std::time::Instant::now();
    let res1 = x.sample(n);
    let time1 = start1.elapsed();
    println!("Time taken: {:?}", time1);
    let y: Vec<Float> = res1.into_iter().collect();
    println!(
        "y expectation: {:?}",
        y.iter().fold(0.0, |sum, x| sum + x / (n as Float))
    );
    // println!("res1:{:?}",res1);

}
