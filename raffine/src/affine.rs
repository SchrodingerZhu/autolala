use std::ffi::c_void;

use melior::{
    Context, ContextRef,
    ir::{Attribute, AttributeLike},
};
use mlir_sys::{MlirAffineExpr, MlirStringRef};

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct AffineExpr<'a>(
    mlir_sys::MlirAffineExpr,
    std::marker::PhantomData<*mut &'a ()>,
);

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

impl std::fmt::Display for AffineExpr<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct FmtCtx<'a, 'b> {
            fmt: &'a mut std::fmt::Formatter<'b>,
            result: std::fmt::Result,
        }
        unsafe extern "C" fn mlir_string_callback(
            string_ref: MlirStringRef,
            user_data: *mut c_void,
        ) {
            let fmt: &mut FmtCtx = unsafe { &mut *(user_data as *mut _) };
            let length = string_ref.length;
            let data = string_ref.data as *const u8;
            let slice = unsafe { std::slice::from_raw_parts(data, length) };
            let chars = std::str::from_utf8(slice).unwrap();
            fmt.result = fmt.result.and_then(|_| fmt.fmt.write_str(chars));
        }
        let mut ctx = FmtCtx {
            fmt: f,
            result: Ok(()),
        };
        let user_data = &mut ctx as *mut _ as *mut c_void;
        unsafe {
            mlir_sys::mlirAffineExprPrint(self.0, Some(mlir_string_callback), user_data);
        }
        ctx.result
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
    /// ## Safety
    ///
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
        let ctx_ref = unsafe { ContextRef::from_raw(mlir_sys::mlirAffineExprGetContext(self.0)) };
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

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct AffineMap<'a>(
    mlir_sys::MlirAffineMap,
    std::marker::PhantomData<*mut &'a ()>,
);

impl<'a> AffineMap<'a> {
    pub fn get_raw(&self) -> mlir_sys::MlirAffineMap {
        self.0
    }
    pub fn context(&self) -> &'_ Context {
        let ctx_ref = unsafe { ContextRef::from_raw(mlir_sys::mlirAffineMapGetContext(self.0)) };
        unsafe { ctx_ref.to_ref() }
    }
    pub fn new(
        ctx: &'a Context,
        num_dims: usize,
        num_symbols: usize,
        exprs: &[AffineExpr<'a>],
    ) -> Self {
        let num_exprs = exprs.len();
        let exprs_ptr: *mut MlirAffineExpr = exprs.as_ptr() as *mut MlirAffineExpr;
        let map = unsafe {
            mlir_sys::mlirAffineMapGet(
                ctx.to_raw(),
                num_dims as isize,
                num_symbols as isize,
                num_exprs as isize,
                exprs_ptr,
            )
        };
        AffineMap(map, std::marker::PhantomData)
    }
    pub fn dump(&self) {
        unsafe { mlir_sys::mlirAffineMapDump(self.0) }
    }
    pub fn to_attr(&self) -> Attribute<'a> {
        let attr = unsafe { mlir_sys::mlirAffineMapAttrGet(self.0) };
        unsafe { Attribute::from_raw(attr) }
    }
    pub fn is_empty(&self) -> bool {
        unsafe { mlir_sys::mlirAffineMapIsEmpty(self.0) }
    }
    pub fn replace(
        &self,
        expr: AffineExpr<'a>,
        replacement: AffineExpr<'a>,
        num_dims: usize,
        num_symbols: usize,
    ) -> Self {
        let map = unsafe {
            mlir_sys::mlirAffineMapReplace(
                self.0,
                expr.0,
                replacement.0,
                num_dims as isize,
                num_symbols as isize,
            )
        };
        AffineMap(map, std::marker::PhantomData)
    }
    pub fn new_empty(ctx: &'a Context) -> Self {
        let map = unsafe { mlir_sys::mlirAffineMapEmptyGet(ctx.to_raw()) };
        AffineMap(map, std::marker::PhantomData)
    }
    pub fn get_result_expr(&self, pos: isize) -> AffineExpr<'a> {
        let expr = unsafe { mlir_sys::mlirAffineMapGetResult(self.0, pos) };
        AffineExpr(expr, std::marker::PhantomData)
    }
    pub fn get_submap(&self, positions: &[isize]) -> AffineMap<'a> {
        let num_positions = positions.len();
        let positions_ptr: *mut isize = positions.as_ptr() as *mut isize;
        let map = unsafe {
            mlir_sys::mlirAffineMapGetSubMap(self.0, num_positions as isize, positions_ptr)
        };
        AffineMap(map, std::marker::PhantomData)
    }
    pub fn num_dims(&self) -> usize {
        unsafe { mlir_sys::mlirAffineMapGetNumDims(self.0) as usize }
    }
    pub fn is_identity(&self) -> bool {
        unsafe { mlir_sys::mlirAffineMapIsIdentity(self.0) }
    }
    pub fn new_constant(ctx: &'a Context, value: i64) -> Self {
        let map = unsafe { mlir_sys::mlirAffineMapConstantGet(ctx.to_raw(), value) };
        AffineMap(map, std::marker::PhantomData)
    }
    pub fn from_attr(attr: Attribute<'a>) -> Option<Self> {
        if !unsafe { mlir_sys::mlirAttributeIsAAffineMap(attr.to_raw()) } {
            return None;
        }
        let map = unsafe { mlir_sys::mlirAffineMapAttrGetValue(attr.to_raw()) };
        Some(AffineMap(map, std::marker::PhantomData))
    }
    pub fn num_inputs(&self) -> usize {
        unsafe { mlir_sys::mlirAffineMapGetNumInputs(self.0) as usize }
    }
    pub fn num_results(&self) -> usize {
        unsafe { mlir_sys::mlirAffineMapGetNumResults(self.0) as usize }
    }
    pub fn num_symbols(&self) -> usize {
        unsafe { mlir_sys::mlirAffineMapGetNumSymbols(self.0) as usize }
    }
    pub fn is_permutation(&self) -> bool {
        unsafe { mlir_sys::mlirAffineMapIsPermutation(self.0) }
    }
    pub fn new_zero_result(ctx: &'a Context, num_dims: usize, num_symbols: usize) -> Self {
        let map = unsafe {
            mlir_sys::mlirAffineMapZeroResultGet(
                ctx.to_raw(),
                num_dims as isize,
                num_symbols as isize,
            )
        };
        AffineMap(map, std::marker::PhantomData)
    }
    pub fn major_submap(&self, num_results: usize) -> Self {
        let map = unsafe { mlir_sys::mlirAffineMapGetMajorSubMap(self.0, num_results as isize) };
        AffineMap(map, std::marker::PhantomData)
    }
    pub fn minor_submap(&self, num_results: usize) -> Self {
        let map = unsafe { mlir_sys::mlirAffineMapGetMinorSubMap(self.0, num_results as isize) };
        AffineMap(map, std::marker::PhantomData)
    }
    pub fn new_permutation(ctx: &'a Context, numbers: &[u32]) -> Self {
        let num_numbers = numbers.len();
        let numbers_ptr: *mut u32 = numbers.as_ptr() as *mut u32;
        let map = unsafe {
            mlir_sys::mlirAffineMapPermutationGet(ctx.to_raw(), num_numbers as isize, numbers_ptr)
        };
        AffineMap(map, std::marker::PhantomData)
    }
    pub fn is_minor_identity(&self) -> bool {
        unsafe { mlir_sys::mlirAffineMapIsMinorIdentity(self.0) }
    }
    pub fn is_single_constant(&self) -> bool {
        unsafe { mlir_sys::mlirAffineMapIsSingleConstant(self.0) }
    }
    pub fn new_minor_identity(ctx: &'a Context, num_dims: usize, num_results: usize) -> Self {
        let map = unsafe {
            mlir_sys::mlirAffineMapMinorIdentityGet(
                ctx.to_raw(),
                num_dims as isize,
                num_results as isize,
            )
        };
        AffineMap(map, std::marker::PhantomData)
    }
    pub fn new_multi_dim_identity(ctx: &'a Context, num_dims: usize) -> Self {
        let map =
            unsafe { mlir_sys::mlirAffineMapMultiDimIdentityGet(ctx.to_raw(), num_dims as isize) };
        AffineMap(map, std::marker::PhantomData)
    }
    pub fn is_projected_permutation(&self) -> bool {
        unsafe { mlir_sys::mlirAffineMapIsProjectedPermutation(self.0) }
    }
}

impl PartialEq for AffineMap<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlir_sys::mlirAffineMapEqual(self.0, other.0) }
    }
}

impl Eq for AffineMap<'_> {}

impl std::fmt::Display for AffineMap<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct FmtCtx<'a, 'b> {
            fmt: &'a mut std::fmt::Formatter<'b>,
            result: std::fmt::Result,
        }
        unsafe extern "C" fn mlir_string_callback(
            string_ref: MlirStringRef,
            user_data: *mut c_void,
        ) {
            let fmt: &mut FmtCtx = unsafe { &mut *(user_data as *mut _) };
            let length = string_ref.length;
            let data = string_ref.data as *const u8;
            let slice = unsafe { std::slice::from_raw_parts(data, length) };
            let chars = std::str::from_utf8(slice).unwrap();
            fmt.result = fmt.result.and_then(|_| fmt.fmt.write_str(chars));
        }
        let mut ctx = FmtCtx {
            fmt: f,
            result: Ok(()),
        };
        let user_data = &mut ctx as *mut _ as *mut c_void;
        unsafe {
            mlir_sys::mlirAffineMapPrint(self.0, Some(mlir_string_callback), user_data);
        }
        ctx.result
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct IntegerSet<'a>(
    mlir_sys::MlirIntegerSet,
    std::marker::PhantomData<*mut &'a ()>,
);

impl<'a> IntegerSet<'a> {
    pub fn new<I>(ctx: &'a Context, num_dims: usize, num_symbols: usize, constraints: I) -> Self
    where
        I: IntoIterator<Item = (AffineExpr<'a>, bool)>,
    {
        let mut exprs = Vec::new();
        let mut eq_flags = Vec::new();
        for (expr, eq) in constraints {
            exprs.push(expr);
            eq_flags.push(eq);
        }
        let num_constraints = exprs.len();
        let exprs_ptr: *mut MlirAffineExpr = exprs.as_ptr() as *mut MlirAffineExpr;
        let eq_flags_ptr: *mut bool = eq_flags.as_ptr() as *mut bool;
        let set = unsafe {
            mlir_sys::mlirIntegerSetGet(
                ctx.to_raw(),
                num_dims as isize,
                num_symbols as isize,
                num_constraints as isize,
                exprs_ptr,
                eq_flags_ptr,
            )
        };
        IntegerSet(set, std::marker::PhantomData)
    }
    pub fn dump(&self) {
        unsafe { mlir_sys::mlirIntegerSetDump(self.0) }
    }
    pub fn to_attr(&self) -> Attribute<'a> {
        let attr = unsafe { mlir_sys::mlirIntegerSetAttrGet(self.0) };
        unsafe { Attribute::from_raw(attr) }
    }
    pub fn context(&self) -> &'_ Context {
        let ctx_ref = unsafe { ContextRef::from_raw(mlir_sys::mlirIntegerSetGetContext(self.0)) };
        unsafe { ctx_ref.to_ref() }
    }
    pub fn new_empty(ctx: &'a Context, num_dims: usize, num_symbols: usize) -> Self {
        let set = unsafe {
            mlir_sys::mlirIntegerSetEmptyGet(ctx.to_raw(), num_dims as isize, num_symbols as isize)
        };
        IntegerSet(set, std::marker::PhantomData)
    }
    pub fn num_dims(&self) -> usize {
        unsafe { mlir_sys::mlirIntegerSetGetNumDims(self.0) as usize }
    }
    pub fn num_symbols(&self) -> usize {
        unsafe { mlir_sys::mlirIntegerSetGetNumSymbols(self.0) as usize }
    }
    pub fn replace(
        &self,
        dim_replacements: &[AffineExpr<'a>],
        sym_replacements: &[AffineExpr<'a>],
        num_dims: usize,
        num_symbols: usize,
    ) -> Self {
        let dim_replacements_ptr = dim_replacements.as_ptr() as *const MlirAffineExpr;
        let sym_replacements_ptr = sym_replacements.as_ptr() as *const MlirAffineExpr;
        let set = unsafe {
            mlir_sys::mlirIntegerSetReplaceGet(
                self.0,
                dim_replacements_ptr,
                sym_replacements_ptr,
                num_dims as isize,
                num_symbols as isize,
            )
        };
        IntegerSet(set, std::marker::PhantomData)
    }
    pub fn from_attr(attr: Attribute<'a>) -> Option<Self> {
        if !unsafe { mlir_sys::mlirAttributeIsAIntegerSet(attr.to_raw()) } {
            return None;
        }
        let set = unsafe { mlir_sys::mlirIntegerSetAttrGetValue(attr.to_raw()) };
        Some(IntegerSet(set, std::marker::PhantomData))
    }
    pub fn num_inputs(&self) -> usize {
        unsafe { mlir_sys::mlirIntegerSetGetNumInputs(self.0) as usize }
    }
    pub fn get_constraint(&self, pos: isize) -> AffineExpr<'a> {
        let expr = unsafe { mlir_sys::mlirIntegerSetGetConstraint(self.0, pos) };
        AffineExpr(expr, std::marker::PhantomData)
    }
    pub fn is_constraint_equal(&self, pos: isize) -> bool {
        unsafe { mlir_sys::mlirIntegerSetIsConstraintEq(self.0, pos) }
    }
    pub fn num_equalities(&self) -> usize {
        unsafe { mlir_sys::mlirIntegerSetGetNumEqualities(self.0) as usize }
    }
    pub fn is_canonical_empty(&self) -> bool {
        unsafe { mlir_sys::mlirIntegerSetIsCanonicalEmpty(self.0) }
    }
    pub fn num_constraints(&self) -> usize {
        unsafe { mlir_sys::mlirIntegerSetGetNumConstraints(self.0) as usize }
    }
    pub fn num_ineqalities(&self) -> usize {
        unsafe { mlir_sys::mlirIntegerSetGetNumInequalities(self.0) as usize }
    }
}

impl PartialEq for IntegerSet<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlir_sys::mlirIntegerSetEqual(self.0, other.0) }
    }
}

impl Eq for IntegerSet<'_> {}

impl std::fmt::Display for IntegerSet<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct FmtCtx<'a, 'b> {
            fmt: &'a mut std::fmt::Formatter<'b>,
            result: std::fmt::Result,
        }
        unsafe extern "C" fn mlir_string_callback(
            string_ref: MlirStringRef,
            user_data: *mut c_void,
        ) {
            let fmt: &mut FmtCtx = unsafe { &mut *(user_data as *mut _) };
            let length = string_ref.length;
            let data = string_ref.data as *const u8;
            let slice = unsafe { std::slice::from_raw_parts(data, length) };
            let chars = std::str::from_utf8(slice).unwrap();
            fmt.result = fmt.result.and_then(|_| fmt.fmt.write_str(chars));
        }
        let mut ctx = FmtCtx {
            fmt: f,
            result: Ok(()),
        };
        let user_data = &mut ctx as *mut _ as *mut c_void;
        unsafe {
            mlir_sys::mlirIntegerSetPrint(self.0, Some(mlir_string_callback), user_data);
        }
        ctx.result
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
        println!("expr4: {}", expr4);
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

    #[test]
    fn test_affine_map_trivial() {
        let context = Context::new();
        let expr1 = AffineExpr::new_constant(&context, 5);
        let expr2 = AffineExpr::new_constant(&context, 3);
        let expr3 = expr1 + expr2;
        let map = AffineMap::new(&context, 0, 0, &[expr3]);
        assert!(!map.is_identity());
        assert!(!map.is_empty());
        assert_eq!(map.num_dims(), 0);
        assert_eq!(map.num_symbols(), 0);
        assert_eq!(map.num_results(), 1);
        assert!(!map.is_permutation());
        assert!(!map.is_minor_identity());
        assert!(map.is_single_constant());
        println!("map: {}", map);
    }
}
