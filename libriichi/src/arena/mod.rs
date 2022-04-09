mod board;
mod game;
mod one_vs_three;
mod result;
mod two_vs_two;

pub use board::Board;
pub use result::{GameResult, KyokuEndState};

use self::one_vs_three::OneVsThree;
use self::two_vs_two::TwoVsTwo;
use crate::py_helper::add_submodule;

use pyo3::prelude::*;

pub(crate) fn register_module(py: Python, prefix: &str, super_mod: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "arena")?;
    m.add_class::<OneVsThree>()?;
    m.add_class::<TwoVsTwo>()?;
    add_submodule(py, prefix, super_mod, m)
}
