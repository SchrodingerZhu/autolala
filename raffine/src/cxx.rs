use crate::affine::AffineMap;
use melior::ir::OperationRef;

#[repr(transparent)]
struct MlirAffineMap(mlir_sys::MlirAffineMap);

#[repr(transparent)]
struct MlirOperation(mlir_sys::MlirOperation);

unsafe impl cxx::ExternType for MlirAffineMap {
    type Id = cxx::type_id!("MlirAffineMap");
    type Kind = cxx::kind::Trivial;
}
unsafe impl cxx::ExternType for MlirOperation {
    type Id = cxx::type_id!("MlirOperation");
    type Kind = cxx::kind::Trivial;
}

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("affine_helper.hpp");
        type MlirAffineMap = super::MlirAffineMap;
        type MlirOperation = super::MlirOperation;

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
