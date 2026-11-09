//! The functions of `litellm.rust_bridge.host.model_cost`, the only Python this module calls. Each is called with the
//! GIL held and no lock of `NativeModelCostMap` held, since the host's lookup reaches `find_key` on it

use pyo3::call::PyCallArgs;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyTuple};

pub(crate) const MODULE: &str = "litellm.rust_bridge.host.model_cost";

/// Where a registration lands in the published table and the row it starts from
#[derive(FromPyObject)]
pub(crate) struct Landing<'py> {
    #[pyo3(attribute)]
    pub(crate) key: String,
    #[pyo3(attribute)]
    pub(crate) base: Bound<'py, PyAny>,
    #[pyo3(attribute)]
    pub(crate) matched_builtin: bool,
}

fn call<'py>(
    py: Python<'py>,
    name: &str,
    args: impl PyCallArgs<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import(MODULE)?.getattr(name)?.call1(args)
}

/// The row object every name in `names` shares, the owner first
pub(crate) fn entry<'py>(
    fields: Bound<'py, PyAny>,
    names: &[String],
) -> PyResult<Bound<'py, PyAny>> {
    let py = fields.py();
    call(
        py,
        "entry",
        (fields, PyTuple::new(py, names)?, PyFrozenSet::empty(py)?),
    )
}

/// Where `registration` lands in `rows`, `None` when `rows` is no longer the published table
pub(crate) fn resolve<'py>(
    rows: &Bound<'py, PyDict>,
    key: &str,
    registration: &Bound<'py, PyAny>,
    custom_llm_provider: Option<&str>,
) -> PyResult<Option<Landing<'py>>> {
    call(
        rows.py(),
        "resolve",
        (rows, key, registration, custom_llm_provider),
    )?
    .extract()
}

/// The row `registration` writes over `current`
pub(crate) fn merged<'py>(
    current: Option<&Bound<'py, PyAny>>,
    landing: &Landing<'py>,
    registration: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    call(
        registration.py(),
        "merged",
        (current, &landing.base, registration),
    )
}

/// The registration kept for a key: `registration` merged into the one kept before it
pub(crate) fn kept<'py>(
    previous: Option<&Bound<'py, PyDict>>,
    registration: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    Ok(call(registration.py(), "kept", (previous, registration))?.cast_into::<PyDict>()?)
}

/// Drops what the host derived from the table after a write landed in it
pub(crate) fn changed(py: Python<'_>) -> PyResult<()> {
    call(py, "changed", ()).map(drop)
}
