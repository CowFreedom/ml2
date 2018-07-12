extern crate rand;

use Float;
use core::{Distribution, PDF};
extern crate crossbeam;


#[doc = "Numerically evaluate the value of an integral using user-defined probability distributions. Is guaranteed to work asymptotically,
if the distributions' support equals the integration domain."]
#[allow(non_snake_case)]
#[inline]
pub fn MultipleImportanceSampling<D,F>(dist:&[D],n:&[usize], f:&F)-> Float
    where 
    D:Distribution<T=Float>+PDF+Sync,
    F:Fn(Float)->Float+Sync

    {
        let m=dist.len(); //number of functions
        let mut val=vec![0.0;m];
        let nThreads=8;
        let chunksize=Float::ceil(m as Float/nThreads as Float) as usize;

        let mut thread_handles=Vec::with_capacity(nThreads);

        crossbeam::scope(|scope|{
            let val_chunks=val.chunks_mut(chunksize);
            let worklists=dist.chunks(chunksize);
            
            for (worklist,v) in worklists.zip(val_chunks){

                thread_handles.push(scope.spawn(move || {                    
                    for (i,p) in worklist.iter().enumerate(){ //i: index, p: probability function                
                        let mut sum=0.0;
                        for j in 0..n[i]{
                            let s=p.sample(&mut rand::thread_rng());
                            let x=f(s);
                                let w=dist.iter().enumerate().fold(0.0,|acc, (iter,x)|{
                                    acc+(n[iter] as Float*x.pdf(&s))
                                });
                            sum=sum+x/w;                 
                        }
                        v[i]=sum;                   
                    }                  
                }));
        }
        }
        );
    

       // println!("Threadhandlenumber: {}",thread_handles.len());
        for handle in thread_handles{
            handle.join();
        }

        val.iter().fold(0.0,|acc,x|{
            acc+x
        })
  
}

#[doc = "Numerically evaluate the value of an integral using user-defined probability distributions. Is guaranteed to work asymptotically,
if the distributions' support equals the integration domain. Algorithm is not explicitely parallel."]
pub fn MultipleImportanceSamplingSingleCore<D,F>(dist:&[D],n:&[usize], f:&F)-> Float
    where 
    D:Distribution<T=Float>+PDF,
    F:Fn(Float)->Float
    {
        let m=dist.len(); //number of functions
        let mut val=vec![0.0;m];

        for (i,p) in dist.iter().enumerate(){ //i: index, p: probability function
            let mut sum=0.0;
            for j in 0..n[i]{
                let s=p.sample(&mut rand::thread_rng());
                let x=f(s);
                    let w=dist.iter().enumerate().fold(0.0,|acc, (iter,x)|{
                        acc+(n[iter] as Float*x.pdf(&s))
                    });
                sum=sum+x/w;
            
            }
            val[i]=sum;
        }

        val.iter().fold(0.0,|acc,x|{
            acc+x
        })
  
}