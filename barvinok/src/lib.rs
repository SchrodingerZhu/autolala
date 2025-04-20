use std::ptr::NonNull;

pub mod qpolynomial;
pub mod space;
pub mod stat;
pub mod value;

#[repr(transparent)]
pub struct Context(NonNull<barvinok_sys::isl_ctx>);

impl Context {
    pub fn new() -> Option<Self> {
        let ctx = unsafe { barvinok_sys::isl_ctx_alloc() };
        let ctx = NonNull::new(ctx)?;
        Some(Self(ctx))
    }
    pub fn set_max_operations(&self, max_operations: usize) {
        unsafe { barvinok_sys::isl_ctx_set_max_operations(self.0.as_ptr(), max_operations as u64) }
    }
    pub fn reset_operations(&self) {
        unsafe { barvinok_sys::isl_ctx_reset_operations(self.0.as_ptr()) }
    }
}
