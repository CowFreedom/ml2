extern crate rand;
use self::rand::{FromEntropy, Rng};
use self::rand::distributions::{Uniform,StandardNormal};
use self::rand::rngs::{SmallRng};
use Float;
use ::core::{Distribution, PDF};
use ::core;
use std::f64::consts;


pub fn RejectionSampleNormal(){
    
    struct SampleNormal{

    }
    
    impl Distribution for SampleNormal{
        type T=Float;
        fn sample<R:Rng+?Sized>(&self, rng:&mut R)->Float{

            let val: f64 = SmallRng::from_entropy().sample(StandardNormal);
            return val;
        }
    }
    let mut rng=SmallRng::from_entropy();
    let Normal=SampleNormal{};
    let normalPdf=|x:&Float|{
        0.3989* f64::powf(2.71,-0.5*x*x)
    };
    let f=|x:&Float|{
        let c:Float=2.0;
        let mut rng=SmallRng::from_entropy();
        let u = rng.gen::<f64>()  as Float;
        
    //    let u: f64 = rng().gen_range(0.0, c*normalPdf(&x));
        if u*c*normalPdf(&x)<normalPdf(&x){
            true
        }
        else {
            false
        }
    };
    let mut samples=Vec::<Float>::with_capacity(20);
    for i in 0..20{
        println!("Iter:{}",i);
        samples.push(core::samplers::RejectionSampleRegion(&Normal,&f));
    }
    println!("{:?}",samples);
    assert_eq!(1,1,"{:?}",samples); 
}

#[allow(non_snake_case)]
#[test]
pub fn RejectionSampleCircle(){
    
    struct EnvelopeShape{
        r:Float
    }
    
    impl Distribution for EnvelopeShape{
        type T=(Float,Float);
        fn sample<R:Rng+?Sized>(&self, rng:&mut R)->(Float,Float){

            let x = rng.gen::<f64>() as Float *self.r;
            let y=  rng.gen::<f64>() as Float *self.r;
            (x,y)
        }
    }
    let radius:Float=1.0;

    let f=|(x,y):&(Float,Float)|{
        if x*x+y*y<radius{
            true
        }
        else {
            false
        }
    };

    let circle=EnvelopeShape{r:5.0};

    let mut samples=Vec::<(Float,Float)>::with_capacity(20);
    for i in 0..20{
        samples.push(core::samplers::RejectionSampleRegion(&circle,&f));
    }

}

#[allow(non_snake_case)]
#[test]
pub fn ImportanceSampleExponential(){
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
  

    let q=[Distributions::Normal, Distributions::Uniform]; //HIER SIND PROBLEME

    //Number of samples per sample distribution
    let n=[100000,100000];
    //Function to evaluate: s: sample, d:destination
     let f=|s:Float|->Float
    {
        f64::powf(consts::E,-1.0*s*s)
    };   

    let x=core::integrators::MultipleImportanceSampling(&q,&n,&f);
   // let x=core::integrators::MultipleImportanceSamplingSingleCore(&q,&n,&f);
   // println!("Result: {:?}",x);

}


#[allow(non_snake_case)]
#[test]
pub fn MetropolisSampling(){
    enum Distributions {
        Normal,
        Uniform
    }
//HIER SIND PROBLEME
    impl Distribution for Distributions{
        type T=[Float;1];
        fn sample<'a,R:Rng+?Sized>(&'a self, rng:&mut R)->Self::T{
            match self{
                Distributions::Normal=>[SmallRng::from_entropy().sample(StandardNormal)],
                Distributions::Uniform=>[4.0*rng.gen::<f64>()-2 as Float]
            } 
        }
    }
    impl PDF for Distributions{
        fn pdf(&self,x:&Self::T)->Float{
            match self{
                Distributions::Normal=>0.3989*f64::powf(consts::E,-0.5*x[0]*x[0]),
                Distributions::Uniform=>{
                    if x[0]>=-2.0 && x[0]<=2.0{
                0.25
            }
            else{
                0.0
            }}
            } 
        }
    }
  

    let q=[Distributions::Normal, Distributions::Uniform]; //HIER SIND PROBLEME


    //Function to evaluate which is proportional to a valid pdf
     let f=|s:&[Float]|->Float
    {
        f64::powf(consts::E,-0.5*s[0]*s[0])
    };   

    let x=core::samplers::MetropolisHastings::new(f,Distributions::Normal,Distributions::Normal);
    x.sample(1);
    //println!("Result: {:?}",x);

}