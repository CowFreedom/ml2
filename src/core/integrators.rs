extern crate rand;

use Float;
use core::{Distribution, PDF};
use std;

//pub trait NewTrait:Distribution<T=Float>+PDF{}
#[allow(non_snake_case)]
#[inline]
pub fn MultipleImportanceSampling<D>(dist:&[&D],n:&[u64], f:&Fn(&Float)->Float)-> Float
    where 
    D:Distribution<T=Float>+PDF,
   // D::T:std::ops::Div,
   // Float:std::convert::From<D::T>

    {
        let k=n.len(); //Number of functions
        let mut allSamples=Vec::with_capacity(k);

        for i in 0..k{
            let mut functionSamples=Vec::with_capacity(n[i] as usize);
            let x=&dist[i];
            
                println!("HEY{}",i);

            for j in 0..n[i]{
                let s=x.sample(&mut rand::thread_rng());
                //println!("S{}:{}",j,s);
                let weight=n[i] as Float*dist[i].pdf(&s)*(1.0/dist.into_iter().enumerate().fold(0.0,|sum,(j,f)| sum+(n[j] as Float)*f.pdf(&s)));    
               //let weight=dist.into_iter().enumerate().fold(0.0,|sum,(i,f)| {println!("n:{:?}",n);0.0});      
                functionSamples.push(f(&s)*(weight/x.pdf(&s)));
            }
            allSamples.push(functionSamples);
        }
        //println!("{:?}",allSamples);
//1.0/(n[0] as Float)*
        1.0/(n[1] as Float)*allSamples[1].iter().fold(0.0,|sum,x| sum+x)  //Hier arbeiten; Bis jetzt wird nur eine Funktion ausgewertet      
}