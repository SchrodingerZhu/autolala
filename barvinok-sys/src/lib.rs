#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::{isl_ctx_alloc, isl_ctx_free};

    #[test]
    fn it_creates_isl_context () {
        unsafe {
            let context = isl_ctx_alloc();
            isl_ctx_free(context);
        }
    }
}