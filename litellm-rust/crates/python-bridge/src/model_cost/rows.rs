use litellm_host_python::to_py;
use litellm_model_catalog::{ModelRow, Row, RowShape};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyTuple};

pub(crate) const MODULE: &str = "litellm.rust_bridge.host.model_cost";

/// Builds a dict holding `rows` in order, aliases sharing their owner's object
pub(crate) fn table<'py>(py: Python<'py>, rows: &[Row<'_>]) -> PyResult<Bound<'py, PyDict>> {
    let table = PyDict::new(py);
    for row in rows {
        let value = match &row.shape {
            RowShape::AliasOf(owner) => table
                .get_item(owner)?
                .expect("an owner is served before its aliases"),
            RowShape::Model(model) => model_row(py, model)?,
            RowShape::Section(value) => to_py(py, value)?.into_bound(py),
        };
        table.set_item(row.key, value)?;
    }
    Ok(table)
}

/// A plain dict, or main's `AliasedCostEntry` for a row with aliases
fn model_row<'py>(py: Python<'py>, model: &ModelRow<'_>) -> PyResult<Bound<'py, PyAny>> {
    let fields = to_py(py, model.fields.as_ref())?.into_bound(py);
    if model.names.is_empty() {
        return Ok(fields);
    }
    py.import(MODULE)?.getattr("entry")?.call1((
        fields,
        PyTuple::new(py, model.names)?,
        PyFrozenSet::empty(py)?,
    ))
}

/// Where a registration lands in a host's table and the row it starts from
pub(crate) struct Landing<'py> {
    pub(crate) key: String,
    pub(crate) base: Bound<'py, PyAny>,
    pub(crate) matched_builtin: bool,
}

/// Where `registration` lands in `rows`, or `None` when `rows` is not the table the host publishes, since the host
/// looks keys up only in that one
pub(crate) fn resolve<'py>(
    rows: &Bound<'py, PyDict>,
    key: &str,
    registration: &Bound<'py, PyAny>,
    custom_llm_provider: Option<&str>,
) -> PyResult<Option<Landing<'py>>> {
    let landing = rows.py().import(MODULE)?.getattr("resolve")?.call1((
        rows,
        key,
        registration,
        custom_llm_provider,
    ))?;
    if landing.is_none() {
        return Ok(None);
    }
    Ok(Some(Landing {
        key: landing.getattr("key")?.extract()?,
        base: landing.getattr("base")?,
        matched_builtin: landing.getattr("matched_builtin")?.extract()?,
    }))
}

/// Writes `registration` over the row under `landing.key`, and under every alias name that shares that row object
pub(crate) fn write<'py>(
    rows: &Bound<'py, PyDict>,
    landing: &Landing<'py>,
    registration: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let current = rows.get_item(&landing.key)?;
    let row = rows.py().import(MODULE)?.getattr("merged")?.call1((
        current.as_ref(),
        &landing.base,
        registration,
    ))?;
    let names: Vec<String> = row.getattr("names")?.extract()?;
    let sharing: Vec<String> = names
        .into_iter()
        .filter(|name| {
            rows.get_item(name)
                .ok()
                .flatten()
                .is_some_and(|served| current.as_ref().is_some_and(|current| served.is(current)))
        })
        .collect();
    rows.set_item(&landing.key, &row)?;
    sharing
        .iter()
        .try_for_each(|name| rows.set_item(name, &row))
}

/// Tells the host a write landed in its table, so it drops what it derived from the table
pub(crate) fn changed(py: Python<'_>) -> PyResult<()> {
    py.import(MODULE)?.getattr("changed")?.call0().map(drop)
}
