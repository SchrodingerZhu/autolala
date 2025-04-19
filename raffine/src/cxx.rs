use crate::affine::{AffineMap, IntegerSet};
use melior::ir::{BlockRef, OperationRef};

#[repr(transparent)]
struct MlirAffineMap(mlir_sys::MlirAffineMap);

#[repr(transparent)]
struct MlirOperation(mlir_sys::MlirOperation);

#[repr(transparent)]
struct MlirIntegerSet(mlir_sys::MlirIntegerSet);

#[repr(transparent)]
struct MlirBlock(mlir_sys::MlirBlock);

unsafe impl cxx::ExternType for MlirAffineMap {
    type Id = cxx::type_id!("MlirAffineMap");
    type Kind = cxx::kind::Trivial;
}
unsafe impl cxx::ExternType for MlirOperation {
    type Id = cxx::type_id!("MlirOperation");
    type Kind = cxx::kind::Trivial;
}

unsafe impl cxx::ExternType for MlirIntegerSet {
    type Id = cxx::type_id!("MlirIntegerSet");
    type Kind = cxx::kind::Trivial;
}

unsafe impl cxx::ExternType for MlirBlock {
    type Id = cxx::type_id!("MlirBlock");
    type Kind = cxx::kind::Trivial;
}

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("affine_helper.hpp");
        type MlirAffineMap = super::MlirAffineMap;
        type MlirOperation = super::MlirOperation;
        type MlirIntegerSet = super::MlirIntegerSet;
        type MlirBlock = super::MlirBlock;

        #[namespace = "raffine"]
        #[cxx_name = "forOpGetLowerBoundMap"]
        fn for_op_get_lower_bound_map(for_op: MlirOperation) -> Result<MlirAffineMap>;

        #[namespace = "raffine"]
        #[cxx_name = "forOpGetUpperBoundMap"]
        fn for_op_get_upper_bound_map(for_op: MlirOperation) -> Result<MlirAffineMap>;

        #[namespace = "raffine"]
        #[cxx_name = "forOpGetStep"]
        fn for_op_get_step(for_op: MlirOperation) -> Result<isize>;

        #[namespace = "raffine"]
        #[cxx_name = "loadStoreOpGetAccessId"]
        fn load_store_op_get_access_id(op: MlirOperation) -> Result<usize>;

        #[namespace = "raffine"]
        #[cxx_name = "loadStoreOpGetAccessMap"]
        fn load_store_op_get_access_map(op: MlirOperation) -> Result<MlirAffineMap>;

        #[namespace = "raffine"]
        #[cxx_name = "ifOpGetCondition"]
        fn if_op_get_condition(if_op: MlirOperation) -> Result<MlirIntegerSet>;

        #[namespace = "raffine"]
        #[cxx_name = "ifOpGetThenBlock"]
        fn if_op_get_then_block(if_op: MlirOperation) -> Result<MlirBlock>;

        #[namespace = "raffine"]
        #[cxx_name = "ifOpGetElseBlock"]
        fn if_op_get_else_block(if_op: MlirOperation) -> Result<MlirBlock>;
    }
}

pub(crate) fn for_op_get_lower_bound_map<'a>(
    for_op: OperationRef<'a, '_>,
) -> Result<AffineMap<'a>, crate::Error> {
    let for_op = MlirOperation(for_op.to_raw());
    let map = ffi::for_op_get_lower_bound_map(for_op)?;
    Ok(unsafe { std::mem::transmute::<MlirAffineMap, AffineMap>(map) })
}

pub(crate) fn for_op_get_upper_bound_map<'a>(
    for_op: OperationRef<'a, '_>,
) -> Result<AffineMap<'a>, crate::Error> {
    let for_op = MlirOperation(for_op.to_raw());
    let map = ffi::for_op_get_upper_bound_map(for_op)?;
    Ok(unsafe { std::mem::transmute::<MlirAffineMap, AffineMap>(map) })
}

pub(crate) fn for_op_get_step(for_op: OperationRef) -> Result<isize, crate::Error> {
    ffi::for_op_get_step(MlirOperation(for_op.to_raw())).map_err(Into::into)
}

pub(crate) fn load_store_op_get_access_id(op: OperationRef) -> Result<usize, crate::Error> {
    ffi::load_store_op_get_access_id(MlirOperation(op.to_raw())).map_err(Into::into)
}

pub(crate) fn load_store_op_get_access_map<'a>(
    op: OperationRef<'a, '_>,
) -> Result<AffineMap<'a>, crate::Error> {
    let op = MlirOperation(op.to_raw());
    let map = ffi::load_store_op_get_access_map(op)?;
    Ok(unsafe { std::mem::transmute::<MlirAffineMap, AffineMap>(map) })
}

pub(crate) fn if_op_get_condition<'a>(
    if_op: OperationRef<'a, '_>,
) -> Result<IntegerSet<'a>, crate::Error> {
    let set = ffi::if_op_get_condition(MlirOperation(if_op.to_raw()))?;
    Ok(unsafe { std::mem::transmute::<MlirIntegerSet, IntegerSet>(set) })
}

pub(crate) fn if_op_get_then_block<'a, 'b>(
    if_op: OperationRef<'a, 'b>,
) -> Result<BlockRef<'a, 'b>, crate::Error> {
    let block = ffi::if_op_get_then_block(MlirOperation(if_op.to_raw()))?;
    Ok(unsafe { BlockRef::from_raw(block.0) })
}

pub(crate) fn if_op_get_else_block<'a, 'b>(
    if_op: OperationRef<'a, 'b>,
) -> Result<Option<BlockRef<'a, 'b>>, crate::Error> {
    let block = ffi::if_op_get_else_block(MlirOperation(if_op.to_raw()))?;
    if block.0.ptr.is_null() {
        Ok(None)
    } else {
        Ok(Some(unsafe { BlockRef::from_raw(block.0) }))
    }
}
