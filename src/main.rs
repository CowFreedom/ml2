extern crate ml2;
extern crate nalgebra as na;
use na::{U2, U3, Dynamic, MatrixArray, MatrixVec,SVD};
use na::Real;

extern crate rand;
use rand::Rng;


fn main(){

}

/*
fn main(){

let m:na::Matrix<f32,na::U3,na::U4,na::MatrixArray<f32,na::U3,na::U4>> = na::Matrix3x4::new(11.0, 12.0, 13.0, 14.0,
                       21.0, 22.0, 23.0, 24.0,
                       31.0, 32.0, 33.0, 34.0);

let k=m.svd(true, true);
let U=k.u.unwrap();
let V_T=k.v_t.unwrap();
let S=k.singular_values;

let pseudo=k.pseudo_inverse(0.01);
//println!("{:?}",U*V_T);
//println!("{:?}",pseudo);

let x=ml2::core::stat::polynomial_regression(&S,0);
println!("x Wert: {}",x(4.0))

}
*/