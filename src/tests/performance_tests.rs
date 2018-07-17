extern crate rand;
use self::rand::{FromEntropy, Rng};
use self::rand::distributions::{Uniform,StandardNormal};
use self::rand::rngs::{SmallRng};
use Float;
use ::core::{Distribution, PDF};
use ::core;
use std::f64::consts;

use std::time::{SystemTime, UNIX_EPOCH};
use std;



//#[cfg(test)]

pub fn ProfileSamplers(){
        enum Distributions {
        Normal,
        Uniform
    }
//HIER SIND PROBLEME
    impl Distribution for Distributions{
        type T=Float;
        fn sample<'a,R:Rng+?Sized>(&'a self, rng:&mut R)->Self::T{
            match self{
                Distributions::Normal=>SmallRng::from_entropy().sample(StandardNormal),
                Distributions::Uniform=>4.0*rng.gen::<f64>()-2 as Float
            } 
        }
    }
    impl PDF for Distributions{
        fn pdf(&self,x:&Self::T)->Float{
            match self{
                Distributions::Normal=>0.3989*f64::powf(consts::E,-0.5*x*x),
                Distributions::Uniform=>{
                    if *x>=-2.0 && *x<=2.0{
                0.25
            }
            else{
                0.0
            }}
            } 
        }
    }
    let q=[Distributions::Normal, Distributions::Uniform];

    //Number of samples per sample distribution
    let k=1_000_000_000;
    let n=[k,k];

    //Function to evaluate: s: sample, d:destination
     let f=|s:Float|->Float
    {
        //f64::powf(consts::E,-1.0*s*s)
        if s<2.0 && s >-2.0{
            f64::powf(consts::E,s*s*s*s)
        }
        else{
            0.0
        }
    };

    let start1 =  std::time::Instant::now();
        let res1=core::integrators::MultipleImportanceSamplingSingleCore(&q,&n,&f);
    let timeSingleCore = start1.elapsed().as_secs();
        let start2 =std::time::Instant::now(); 
     let res2=core::integrators::MultipleImportanceSampling(&q,&n,&f);       
    let timeMultiCore = start2.elapsed().as_secs();
        println!("SingleCore: {:?}\nMultiCore: {:?}",timeSingleCore,timeMultiCore);
        println!("res1:{}, res2:{}",res1,res2);



}

