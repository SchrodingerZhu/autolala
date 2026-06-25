//! Thin, safe-ish wrappers over the parts of the MLIR C API (`mlir-sys`) that
//! `melior` 0.27 does not expose directly: affine maps/expressions, integer
//! sets, shaped-type shapes, and file/line/column source locations.
//!
//! Every handle borrows the owning [`melior::Context`] for `'c`, mirroring the
//! lifetime discipline melior itself uses. All raw pointers come from a live
//! module owned by that context, so the wrappers only ever read.

use std::marker::PhantomData;

use melior::ir::attribute::AttributeLike;
use melior::ir::operation::OperationLike;
use melior::ir::r#type::TypeLike;
use melior::ir::{Attribute, OperationRef, Type};

/// A 1-based file/line/column source range, as recorded by MLIR locations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceSpan {
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
}

/// Kind of an affine expression node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffineExprKind {
    Add,
    Mul,
    Mod,
    FloorDiv,
    CeilDiv,
    Dim,
    Symbol,
    Constant,
}

/// An `MlirAffineExpr` borrowing the owning context for `'c`.
#[derive(Debug, Clone, Copy)]
pub struct AffineExpr<'c>(mlir_sys::MlirAffineExpr, PhantomData<&'c ()>);

impl<'c> AffineExpr<'c> {
    fn wrap(raw: mlir_sys::MlirAffineExpr) -> Self {
        Self(raw, PhantomData)
    }

    pub fn kind(&self) -> AffineExprKind {
        use mlir_sys as s;
        unsafe {
            if s::mlirAffineExprIsAAdd(self.0) {
                AffineExprKind::Add
            } else if s::mlirAffineExprIsAMul(self.0) {
                AffineExprKind::Mul
            } else if s::mlirAffineExprIsAMod(self.0) {
                AffineExprKind::Mod
            } else if s::mlirAffineExprIsAFloorDiv(self.0) {
                AffineExprKind::FloorDiv
            } else if s::mlirAffineExprIsACeilDiv(self.0) {
                AffineExprKind::CeilDiv
            } else if s::mlirAffineExprIsADim(self.0) {
                AffineExprKind::Dim
            } else if s::mlirAffineExprIsASymbol(self.0) {
                AffineExprKind::Symbol
            } else {
                // Constant is the only remaining primitive kind.
                AffineExprKind::Constant
            }
        }
    }

    pub fn lhs(&self) -> AffineExpr<'c> {
        AffineExpr::wrap(unsafe { mlir_sys::mlirAffineBinaryOpExprGetLHS(self.0) })
    }

    pub fn rhs(&self) -> AffineExpr<'c> {
        AffineExpr::wrap(unsafe { mlir_sys::mlirAffineBinaryOpExprGetRHS(self.0) })
    }

    pub fn constant_value(&self) -> i64 {
        unsafe { mlir_sys::mlirAffineConstantExprGetValue(self.0) }
    }

    /// Position of a dim (`d_k`) or symbol (`s_k`) expression.
    pub fn position(&self) -> usize {
        match self.kind() {
            AffineExprKind::Dim => unsafe {
                mlir_sys::mlirAffineDimExprGetPosition(self.0) as usize
            },
            AffineExprKind::Symbol => unsafe {
                mlir_sys::mlirAffineSymbolExprGetPosition(self.0) as usize
            },
            _ => 0,
        }
    }
}

/// An `MlirAffineMap` borrowing the owning context for `'c`.
#[derive(Debug, Clone, Copy)]
pub struct AffineMap<'c>(mlir_sys::MlirAffineMap, PhantomData<&'c ()>);

impl<'c> AffineMap<'c> {
    /// Extract the affine map carried by an `AffineMapAttr`, if that is what the
    /// attribute is.
    pub fn from_attribute(attr: Attribute<'c>) -> Option<Self> {
        let raw = attr.to_raw();
        if !unsafe { mlir_sys::mlirAttributeIsAAffineMap(raw) } {
            return None;
        }
        Some(Self(
            unsafe { mlir_sys::mlirAffineMapAttrGetValue(raw) },
            PhantomData,
        ))
    }

    pub fn num_dims(&self) -> usize {
        unsafe { mlir_sys::mlirAffineMapGetNumDims(self.0) as usize }
    }

    /// Number of operands the map consumes (`num_dims + num_symbols`).
    pub fn num_inputs(&self) -> usize {
        unsafe { mlir_sys::mlirAffineMapGetNumInputs(self.0) as usize }
    }

    pub fn num_results(&self) -> usize {
        unsafe { mlir_sys::mlirAffineMapGetNumResults(self.0) as usize }
    }

    pub fn result(&self, index: usize) -> AffineExpr<'c> {
        AffineExpr::wrap(unsafe { mlir_sys::mlirAffineMapGetResult(self.0, index as isize) })
    }
}

/// An `MlirIntegerSet` borrowing the owning context for `'c`.
#[derive(Debug, Clone, Copy)]
pub struct IntegerSet<'c>(mlir_sys::MlirIntegerSet, PhantomData<&'c ()>);

impl<'c> IntegerSet<'c> {
    pub fn from_attribute(attr: Attribute<'c>) -> Option<Self> {
        let raw = attr.to_raw();
        if !unsafe { mlir_sys::mlirAttributeIsAIntegerSet(raw) } {
            return None;
        }
        Some(Self(
            unsafe { mlir_sys::mlirIntegerSetAttrGetValue(raw) },
            PhantomData,
        ))
    }

    pub fn num_dims(&self) -> usize {
        unsafe { mlir_sys::mlirIntegerSetGetNumDims(self.0) as usize }
    }

    pub fn num_symbols(&self) -> usize {
        unsafe { mlir_sys::mlirIntegerSetGetNumSymbols(self.0) as usize }
    }

    pub fn num_constraints(&self) -> usize {
        unsafe { mlir_sys::mlirIntegerSetGetNumConstraints(self.0) as usize }
    }

    pub fn constraint(&self, index: usize) -> AffineExpr<'c> {
        AffineExpr::wrap(unsafe { mlir_sys::mlirIntegerSetGetConstraint(self.0, index as isize) })
    }

    /// `true` if constraint `index` is an equality (`expr == 0`), `false` for an
    /// inequality (`expr >= 0`).
    pub fn constraint_is_eq(&self, index: usize) -> bool {
        unsafe { mlir_sys::mlirIntegerSetIsConstraintEq(self.0, index as isize) }
    }
}

/// Read the integer payload of an `IntegerAttr` (e.g. an `affine.for` step).
pub fn integer_attr_value(attr: Attribute<'_>) -> Option<i64> {
    let raw = attr.to_raw();
    if !unsafe { mlir_sys::mlirAttributeIsAInteger(raw) } {
        return None;
    }
    Some(unsafe { mlir_sys::mlirIntegerAttrGetValueInt(raw) })
}

/// Shape of a shaped type (`memref`, `tensor`, ...). `None` entries are dynamic
/// dimensions (`?`).
pub fn shaped_type_shape(ty: Type<'_>) -> Option<Vec<Option<i64>>> {
    let raw = ty.to_raw();
    if !unsafe { mlir_sys::mlirTypeIsAShaped(raw) } {
        return None;
    }
    if !unsafe { mlir_sys::mlirShapedTypeHasRank(raw) } {
        return None;
    }
    let rank = unsafe { mlir_sys::mlirShapedTypeGetRank(raw) };
    let mut dims = Vec::with_capacity(rank.max(0) as usize);
    for i in 0..rank {
        let dim = i as isize;
        if unsafe { mlir_sys::mlirShapedTypeIsDynamicDim(raw, dim) } {
            dims.push(None);
        } else {
            dims.push(Some(unsafe {
                mlir_sys::mlirShapedTypeGetDimSize(raw, dim)
            }));
        }
    }
    Some(dims)
}

/// Resolve an operation's location to a [`SourceSpan`] when it is a concrete
/// file/line/column range. Fused/named/unknown locations yield `None`.
pub fn operation_span(op: OperationRef<'_, '_>) -> Option<SourceSpan> {
    let loc = op.location().to_raw();
    if !unsafe { mlir_sys::mlirLocationIsAFileLineColRange(loc) } {
        return None;
    }
    unsafe {
        Some(SourceSpan {
            start_line: mlir_sys::mlirLocationFileLineColRangeGetStartLine(loc) as u32,
            start_col: mlir_sys::mlirLocationFileLineColRangeGetStartColumn(loc) as u32,
            end_line: mlir_sys::mlirLocationFileLineColRangeGetEndLine(loc) as u32,
            end_col: mlir_sys::mlirLocationFileLineColRangeGetEndColumn(loc) as u32,
        })
    }
}
