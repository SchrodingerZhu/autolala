use crate::affine::{AffineMap, IntegerSet};
use melior::ir::{BlockRef, Module, OperationRef, Value, ValueLike};

#[repr(transparent)]
struct MlirAffineMap(mlir_sys::MlirAffineMap);

#[repr(transparent)]
struct MlirOperation(mlir_sys::MlirOperation);

#[repr(transparent)]
struct MlirIntegerSet(mlir_sys::MlirIntegerSet);

#[repr(transparent)]
struct MlirBlock(mlir_sys::MlirBlock);

#[repr(transparent)]
struct MlirValue(mlir_sys::MlirValue);

#[repr(transparent)]
struct MlirModule(mlir_sys::MlirModule);

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

unsafe impl cxx::ExternType for MlirValue {
    type Id = cxx::type_id!("MlirValue");
    type Kind = cxx::kind::Trivial;
}

unsafe impl cxx::ExternType for MlirModule {
    type Id = cxx::type_id!("MlirModule");
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
        type MlirValue = super::MlirValue;
        type MlirModule = super::MlirModule;

        #[namespace = "mlir"]
        type DominanceInfo;

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

        #[namespace = "raffine"]
        #[cxx_name = "forOpGetLowerBoundOperands"]
        fn for_op_get_lower_bound_operands(for_op: MlirOperation) -> Result<Vec<MlirValue>>;

        #[namespace = "raffine"]
        #[cxx_name = "forOpGetUpperBoundOperands"]
        fn for_op_get_upper_bound_operands(for_op: MlirOperation) -> Result<Vec<MlirValue>>;

        #[namespace = "raffine"]
        #[cxx_name = "forOpGetInductionVar"]
        fn for_op_get_induction_variable(for_op: MlirOperation) -> Result<MlirValue>;

        #[namespace = "raffine"]
        #[cxx_name = "createDominanceInfo"]
        fn create_dominance_info(module: MlirModule) -> UniquePtr<DominanceInfo>;

        #[namespace = "raffine"]
        #[cxx_name = "mlirValueProperlyDominatesOperation"]
        fn value_properly_dominates_operation(
            value: MlirValue,
            operation: MlirOperation,
            dominance_info: &DominanceInfo,
        ) -> bool;

        #[namespace = "raffine"]
        #[cxx_name = "loadStoreOpGetAffineOperands"]
        fn load_store_op_get_affine_operands(target: MlirOperation) -> Result<Vec<MlirValue>>;

        #[namespace = "raffine"]
        #[cxx_name = "ifOpGetConditionOperands"]
        fn if_op_get_condition_operands(if_op: MlirOperation) -> Result<Vec<MlirValue>>;
    }

    impl Vec<MlirValue> {}
}

#[repr(transparent)]
pub struct DominanceInfo<'a>(
    cxx::UniquePtr<ffi::DominanceInfo>,
    std::marker::PhantomData<&'a ()>,
);

impl<'a> DominanceInfo<'a> {
    pub fn new(module: &Module<'a>) -> Self {
        let module = MlirModule(module.to_raw());
        let ptr = ffi::create_dominance_info(module);
        Self(ptr, std::marker::PhantomData)
    }

    pub fn properly_dominates(
        &self,
        value: Value<'a, '_>,
        operation: OperationRef<'a, '_>,
    ) -> bool {
        ffi::value_properly_dominates_operation(
            MlirValue(value.to_raw()),
            MlirOperation(operation.to_raw()),
            &self.0,
        )
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

pub(crate) fn for_op_get_lower_bound_operands<'a, 'b>(
    for_op: OperationRef<'a, 'b>,
) -> Result<Vec<Value<'a, 'b>>, crate::Error> {
    let for_op = MlirOperation(for_op.to_raw());
    let operands = ffi::for_op_get_lower_bound_operands(for_op)?;
    Ok(operands
        .into_iter()
        .map(|v| unsafe { Value::from_raw(v.0) })
        .collect())
}

pub(crate) fn for_op_get_upper_bound_operands<'a, 'b>(
    for_op: OperationRef<'a, 'b>,
) -> Result<Vec<Value<'a, 'b>>, crate::Error> {
    let for_op = MlirOperation(for_op.to_raw());
    let operands = ffi::for_op_get_upper_bound_operands(for_op)?;
    Ok(operands
        .into_iter()
        .map(|v| unsafe { Value::from_raw(v.0) })
        .collect())
}

pub(crate) fn for_op_get_induction_variable<'a, 'b>(
    for_op: OperationRef<'a, 'b>,
) -> Result<Value<'a, 'b>, crate::Error> {
    let for_op = MlirOperation(for_op.to_raw());
    let induction_var = ffi::for_op_get_induction_variable(for_op)?;
    Ok(unsafe { Value::from_raw(induction_var.0) })
}

pub(crate) fn load_store_op_get_affine_operands<'a, 'b>(
    target: OperationRef<'a, 'b>,
) -> Result<Vec<Value<'a, 'b>>, crate::Error> {
    let target = MlirOperation(target.to_raw());
    let operands = ffi::load_store_op_get_affine_operands(target)?;
    Ok(operands
        .into_iter()
        .map(|v| unsafe { Value::from_raw(v.0) })
        .collect())
}

pub(crate) fn if_op_get_condition_operands<'a, 'b>(
    if_op: OperationRef<'a, 'b>,
) -> Result<Vec<Value<'a, 'b>>, crate::Error> {
    let if_op = MlirOperation(if_op.to_raw());
    let operands = ffi::if_op_get_condition_operands(if_op)?;
    Ok(operands
        .into_iter()
        .map(|v| unsafe { Value::from_raw(v.0) })
        .collect())
}
