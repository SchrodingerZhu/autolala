use std::ptr::NonNull;

use num_traits::PrimInt;

use crate::Context;

#[repr(transparent)]
pub struct Value<'a> {
    handle: NonNull<barvinok_sys::isl_val>,
    marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> Value<'a> {
    pub fn new_zero(ctx: &'a Context) -> Option<Self> {
        let handle = unsafe { barvinok_sys::isl_val_zero(ctx.0.as_ptr()) };
        let handle = NonNull::new(handle)?;
        Some(Self {
            handle,
            marker: std::marker::PhantomData,
        })
    }
    pub fn new_one(ctx: &'a Context) -> Option<Self> {
        let handle = unsafe { barvinok_sys::isl_val_one(ctx.0.as_ptr()) };
        let handle = NonNull::new(handle)?;
        Some(Self {
            handle,
            marker: std::marker::PhantomData,
        })
    }
    pub fn new_negone(ctx: &'a Context) -> Option<Self> {
        let handle = unsafe { barvinok_sys::isl_val_negone(ctx.0.as_ptr()) };
        let handle = NonNull::new(handle)?;
        Some(Self {
            handle,
            marker: std::marker::PhantomData,
        })
    }
    pub fn new_nan(ctx: &'a Context) -> Option<Self> {
        let handle = unsafe { barvinok_sys::isl_val_nan(ctx.0.as_ptr()) };
        let handle = NonNull::new(handle)?;
        Some(Self {
            handle,
            marker: std::marker::PhantomData,
        })
    }
    pub fn new_infty(ctx: &'a Context) -> Option<Self> {
        let handle = unsafe { barvinok_sys::isl_val_infty(ctx.0.as_ptr()) };
        let handle = NonNull::new(handle)?;
        Some(Self {
            handle,
            marker: std::marker::PhantomData,
        })
    }
    pub fn new_neg_infty(ctx: &'a Context) -> Option<Self> {
        let handle = unsafe { barvinok_sys::isl_val_neginfty(ctx.0.as_ptr()) };
        let handle = NonNull::new(handle)?;
        Some(Self {
            handle,
            marker: std::marker::PhantomData,
        })
    }
    pub fn new_si(ctx: &'a Context, value: i64) -> Option<Self> {
        let handle = unsafe { barvinok_sys::isl_val_int_from_si(ctx.0.as_ptr(), value) };
        let handle = NonNull::new(handle)?;
        Some(Self {
            handle,
            marker: std::marker::PhantomData,
        })
    }
    pub fn new_ui(ctx: &'a Context, value: u64) -> Option<Self> {
        let handle = unsafe { barvinok_sys::isl_val_int_from_ui(ctx.0.as_ptr(), value) };
        let handle = NonNull::new(handle)?;
        Some(Self {
            handle,
            marker: std::marker::PhantomData,
        })
    }
    pub fn new_chunks<T: PrimInt>(ctx: &'a Context, value: &[T]) -> Option<Self> {
        let handle = unsafe {
            barvinok_sys::isl_val_int_from_chunks(
                ctx.0.as_ptr(),
                value.len(),
                std::mem::size_of::<T>(),
                value.as_ptr() as *const std::ffi::c_void,
            )
        };
        let handle = NonNull::new(handle)?;
        Some(Self {
            handle,
            marker: std::marker::PhantomData,
        })
    }
    pub fn try_clone(&self) -> Option<Self> {
        let handle = unsafe { barvinok_sys::isl_val_copy(self.handle.as_ptr()) };
        let handle = NonNull::new(handle)?;
        Some(Self {
            handle,
            marker: std::marker::PhantomData,
        })
    }
}

impl Drop for Value<'_> {
    fn drop(&mut self) {
        unsafe { barvinok_sys::isl_val_free(self.handle.as_ptr()) };
    }
}

impl Clone for Value<'_> {
    fn clone(&self) -> Self {
        self.try_clone().expect("failed to clone ISL value")
    }
}
