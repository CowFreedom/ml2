/*Check the architecture of the user's machine.
Adjusts floating point length accordingly.*/
#[cfg(target_pointer_width = "64")]
type Float=f64;
#[cfg(target_pointer_width = "32")]
type Float=f32;

pub mod core;
//pub mod tests;

#[cfg(test)]
    mod tests;
