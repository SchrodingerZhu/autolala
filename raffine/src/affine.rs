use std::ffi::c_void;

use melior::{Context, ContextRef};
use mlir_sys::MlirStringRef;

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct AffineExpr<'a>(mlir_sys::MlirAffineExpr, std::marker::PhantomData<*mut &'a ()>);

pub enum AffineExprKind {
    Add,
    Dim,
    Mod,
    Mul,
    Symbol,
    CeilDiv,
    Constant,
    FloorDiv,
}

impl PartialEq for AffineExpr<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlir_sys::mlirAffineExprEqual(self.0, other.0) }
    }
}

impl Eq for AffineExpr<'_> {}

impl ToString for AffineExpr<'_> {
    fn to_string(&self) -> String {
        let mut buffer = String::new();
        unsafe extern "C" fn mlir_string_callback(
            string_ref: MlirStringRef,
            user_data: *mut c_void,
        ){
            let cell = unsafe { &mut *(user_data as *mut String) };
            let length = string_ref.length;
            let data = string_ref.data as *const u8;
            let slice  = unsafe { std::slice::from_raw_parts(data, length) };
            let chars = std::str::from_utf8(slice).unwrap();
            cell.push_str(chars);
        }
        let user_data = &mut buffer as *mut _ as *mut c_void;
        unsafe {
            mlir_sys::mlirAffineExprPrint(
                self.0,
                Some(mlir_string_callback),
                user_data,
            );
        }
        buffer
    }
}

impl<'a> AffineExpr<'a> {
    pub fn get_kind(&self) -> AffineExprKind {
        use mlir_sys as sys;
        unsafe {
            if sys::mlirAffineExprIsAAdd(self.0) {
                AffineExprKind::Add
            } else if sys::mlirAffineExprIsADim(self.0) {
                AffineExprKind::Dim
            } else if sys::mlirAffineExprIsAMod(self.0) {
                AffineExprKind::Mod
            } else if sys::mlirAffineExprIsAMul(self.0) {
                AffineExprKind::Mul
            } else if sys::mlirAffineExprIsASymbol(self.0) {
                AffineExprKind::Symbol
            } else if sys::mlirAffineExprIsACeilDiv(self.0) {
                AffineExprKind::CeilDiv
            } else if sys::mlirAffineExprIsAConstant(self.0) {
                AffineExprKind::Constant
            } else if sys::mlirAffineExprIsAFloorDiv(self.0) {
                AffineExprKind::FloorDiv
            } else {
                panic!("Unknown affine expression kind")
            }
        }
    }
    pub unsafe fn from_raw(_ctx: &'a Context, expr: mlir_sys::MlirAffineExpr) -> Self {
        AffineExpr(expr, std::marker::PhantomData)
    }
    pub fn get_raw(&self) -> mlir_sys::MlirAffineExpr {
        self.0
    }
    pub fn is_pure(&self) -> bool {
        unsafe { mlir_sys::mlirAffineExprIsPureAffine(self.0) }
    }
    pub fn is_binary(&self) -> bool {
        unsafe { mlir_sys::mlirAffineExprIsABinary(self.0) }
    }
    pub fn is_multiple_of(&self, value: i64) -> bool {
        unsafe { mlir_sys::mlirAffineExprIsMultipleOf(self.0, value) }
    }
    pub fn is_symbol_or_constant(&self) -> bool {
        unsafe { mlir_sys::mlirAffineExprIsSymbolicOrConstant(self.0) }
    }
    pub fn get_value(&self) -> Option<i64> {
        if matches!(self.get_kind(), AffineExprKind::Constant) {
            let value = unsafe { mlir_sys::mlirAffineConstantExprGetValue(self.0) };
            Some(value)
        } else {
            None
        }
    }
    pub fn get_position(&self) -> Option<isize> {
        match self.get_kind() {
            AffineExprKind::Dim => {
                let pos = unsafe { mlir_sys::mlirAffineDimExprGetPosition(self.0) };
                Some(pos)
            }
            AffineExprKind::Symbol => {
                let pos = unsafe { mlir_sys::mlirAffineSymbolExprGetPosition(self.0) };
                Some(pos)
            }
            _ => None,
        }
    }
    pub fn context(&self) -> &'a Context {
        let ctx_ref = unsafe { ContextRef::from_raw(mlir_sys::mlirAffineExprGetContext(self.0) )};
        unsafe { ctx_ref.to_ref() }
    }
    pub fn get_lhs(&self) -> Option<AffineExpr<'_>> {
        if self.is_binary() {
            let lhs = unsafe { mlir_sys::mlirAffineBinaryOpExprGetLHS(self.0) };
            Some(unsafe { AffineExpr::from_raw(self.context(), lhs) })
        } else {
            None
        }
    }
    pub fn get_rhs(&self) -> Option<AffineExpr<'_>> {
        if self.is_binary() {
            let rhs = unsafe { mlir_sys::mlirAffineBinaryOpExprGetRHS(self.0) };
            Some(unsafe { AffineExpr::from_raw(self.context(), rhs) })
        } else {
            None
        }
    }
    pub fn new_constant(ctx: &'a Context, value: i64) -> Self {
        let expr = unsafe { mlir_sys::mlirAffineConstantExprGet(ctx.to_raw(), value) };
        unsafe { AffineExpr::from_raw(ctx, expr) }
    }
    pub fn new_dim(ctx: &'a Context, position: isize) -> Self {
        let expr = unsafe { mlir_sys::mlirAffineDimExprGet(ctx.to_raw(), position) };
        unsafe { AffineExpr::from_raw(ctx, expr) }
    }
    pub fn new_symbol(ctx: &'a Context, position: isize) -> Self {
        let expr = unsafe { mlir_sys::mlirAffineSymbolExprGet(ctx.to_raw(), position) };
        unsafe { AffineExpr::from_raw(ctx, expr) }
    }
    pub fn ceil_div(&self, rhs: Self) -> Self {
        let expr = unsafe { mlir_sys::mlirAffineCeilDivExprGet(self.0, rhs.0) };
        unsafe { AffineExpr::from_raw(self.context(), expr) }
    }
    pub fn dump(&self) {
        unsafe { mlir_sys::mlirAffineExprDump(self.0) }
    }
}

impl<'a> std::ops::Add for AffineExpr<'a> {
    type Output = AffineExpr<'a>;

    fn add(self, other: Self) -> Self::Output {
        let ctx = self.context();
        let expr = unsafe { mlir_sys::mlirAffineAddExprGet(self.0, other.0) };
        unsafe { AffineExpr::from_raw(ctx, expr) }
    }
}

impl<'a> std::ops::Mul for AffineExpr<'a> {
    type Output = AffineExpr<'a>;

    fn mul(self, other: Self) -> Self::Output {
        let ctx = self.context();
        let expr = unsafe { mlir_sys::mlirAffineMulExprGet(self.0, other.0) };
        unsafe { AffineExpr::from_raw(ctx, expr) }
    }
}

impl<'a> std::ops::Rem for AffineExpr<'a> {
    type Output = AffineExpr<'a>;

    fn rem(self, other: Self) -> Self::Output {
        let ctx = self.context();
        let expr = unsafe { mlir_sys::mlirAffineModExprGet(self.0, other.0) };
        unsafe { AffineExpr::from_raw(ctx, expr) }
    }
}

impl<'a> std::ops::Div for AffineExpr<'a> {
    type Output = AffineExpr<'a>;

    fn div(self, other: Self) -> Self::Output {
        let ctx = self.context();
        let expr = unsafe { mlir_sys::mlirAffineFloorDivExprGet(self.0, other.0) };
        unsafe { AffineExpr::from_raw(ctx, expr) }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use melior::Context;

    #[test]
    fn test_affine_expr() {
        let context = Context::new();
        let expr1 = AffineExpr::new_constant(&context, 5);
        let expr2 = AffineExpr::new_constant(&context, 3);
        let expr3 = expr1 + expr2;
        assert_eq!(expr3.get_value(), Some(8));
    }

    #[test]
    fn test_affine_expr_with_symbol_and_constant_and_dim() {
        let context = Context::new();
        let expr1 = AffineExpr::new_constant(&context, 5);
        let expr2 = AffineExpr::new_symbol(&context, 0);
        let expr3 = AffineExpr::new_dim(&context, 0);
        let expr4 = expr1 + expr2 + expr3;
        assert_eq!(expr4.get_value(), None);
        assert_eq!(expr4.to_string(), "d0 + s0 + 5");
        println!("expr4: {}", expr4.to_string());
    }
    #[test]
    fn test_affine_expr_mul() {
        let context = Context::new();
        let expr1 = AffineExpr::new_constant(&context, 5);
        let expr2 = AffineExpr::new_constant(&context, 3);
        let expr3 = expr1 * expr2;
        assert_eq!(expr3.get_value(), Some(15));
    }

    #[test]
    fn test_affine_expr_rem() {
        let context = Context::new();
        let expr1 = AffineExpr::new_constant(&context, 5);
        let expr2 = AffineExpr::new_constant(&context, 3);
        let expr3 = expr1 % expr2;
        assert_eq!(expr3.get_value(), Some(2));
    }
    #[test]
    fn test_affine_expr_div() {
        let context = Context::new();
        let expr1 = AffineExpr::new_constant(&context, 5);
        let expr2 = AffineExpr::new_constant(&context, 3);
        let expr3 = expr1 / expr2;
        assert_eq!(expr3.get_value(), Some(1));
    }
    #[test]
    fn test_affine_expr_ceil_div() {
        let context = Context::new();
        let expr1 = AffineExpr::new_constant(&context, 5);
        let expr2 = AffineExpr::new_constant(&context, 3);
        let expr3 = expr1.ceil_div(expr2);
        assert_eq!(expr3.get_value(), Some(2));
    }
}