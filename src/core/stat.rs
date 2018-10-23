extern crate nalgebra as na;
use self::na::{U2, U3, Dynamic, MatrixArray, MatrixVec};

use Float;

type DMat = na::Matrix<Float, Dynamic, Dynamic, MatrixVec<Float, Dynamic, Dynamic>>;
///Calculation of regression polynom via l2 norm. m=number of basis functions.
///
pub fn linear_basis_regression<T>(data: &T, m: u8, basis: &Fn(Float) -> Float) -> Float {
    let Phi = DMat::from_fn(4, 3, |r, c| if r == c { 1.0 } else { 0.0 });

    5.0
}

///Polynomial regression of the target data with m basis functions.
///Returns the regression function which can be used to make predictions
///about future data.
///
pub fn polynomial_regression<T>(data: &T, m: u8) -> Box<Fn(Float) -> Float>
where
    T: ::std::fmt::Debug,
{
    println!("{:?}", data);
    //linear_basis_regression(&data,0,|x| 5.0);
    Box::new(|x| x)
}

fn test(u1: usize, u2: usize) -> Float {
    5.0
}


/*Calculate design matrix Phi. Used for (XXT)^(-1)Xt calculation.*/
fn calc_Phi<T>(data: &T, m: u8, basis: &Fn(Float) -> Float) {
    //let Phi=DMat::new();
}

#[cfg(test)]
pub fn test1<T>(data: T) {


    assert_eq!(1, 1);
}
