use super::PlayerState;
use crate::mjai::Event;
use crate::tile::Tile;
use crate::tuz;

use anyhow::{bail, ensure, Result};
use pyo3::prelude::*;
use pyo3::PyObjectProtocol;
use serde::Serialize;

#[pyclass]
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct ActionCandidate {
    #[pyo3(get)]
    pub can_discard: bool,
    #[pyo3(get)]
    pub can_chi_low: bool,
    #[pyo3(get)]
    pub can_chi_mid: bool,
    #[pyo3(get)]
    pub can_chi_high: bool,
    #[pyo3(get)]
    pub can_pon: bool,
    #[pyo3(get)]
    pub can_daiminkan: bool,
    #[pyo3(get)]
    pub can_kakan: bool,
    #[pyo3(get)]
    pub can_ankan: bool,
    #[pyo3(get)]
    pub can_riichi: bool,
    #[pyo3(get)]
    pub can_tsumo_agari: bool,
    #[pyo3(get)]
    pub can_ron_agari: bool,
    #[pyo3(get)]
    pub can_ryukyoku: bool,

    #[pyo3(get)]
    pub target_actor: u8,
}

#[pyproto]
impl PyObjectProtocol for ActionCandidate {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[pymethods]
impl ActionCandidate {
    #[getter]
    #[inline]
    pub const fn can_chi(&self) -> bool {
        self.can_chi_low || self.can_chi_mid || self.can_chi_high
    }

    #[getter]
    #[inline]
    pub const fn can_act(&self) -> bool {
        self.can_discard
            || self.can_chi()
            || self.can_pon
            || self.can_daiminkan
            || self.can_kakan
            || self.can_ankan
            || self.can_riichi
            || self.can_tsumo_agari
            || self.can_ron_agari
            || self.can_ryukyoku
    }
}

impl PlayerState {
    /// Check if `action` is a valid reaction to the current state.
    pub fn validate_action(&self, action: &Event) -> Result<()> {
        let cans = self.last_cans;

        match action {
            Event::Ryukyoku { .. } => {
                ensure!(cans.can_ryukyoku, "cannot ryukyoku");
                return Ok(());
            }
            Event::None => {
                return Ok(());
            }
            _ => (),
        };

        if let Some(actor) = action.actor() {
            ensure!(
                actor == self.player_id,
                "actor is {actor}, not self ({})",
                self.player_id,
            );
        } else {
            bail!("action does not have actor and is not ryukyoku");
        }

        match *action {
            Event::Dahai { pai, tsumogiri, .. } => {
                ensure!(cans.can_discard, "cannot discard");
                self.ensure_tiles_in_hand(&[pai])?;
                if tsumogiri {
                    if let Some(tile) = self.last_self_tsumo {
                        ensure!(tile == pai, "cannot tsumogiri");
                    } else {
                        bail!("tsumogiri but the player has not dealed any tile yet");
                    }
                }
            }

            Event::Reach { .. } => {
                ensure!(cans.can_riichi, "cannot riichi");
            }

            Event::Chi {
                actor,
                target,
                pai,
                consumed,
            } => {
                ensure!((target + 1) % 4 == actor, "chi from non-kamicha");
                ensure!(
                    self.last_kawa_tile.filter(|&tile| tile == pai).is_some(),
                    "chi target is not the last kawa tile",
                );
                self.ensure_tiles_in_hand(&consumed)?;

                let low = consumed[0].deaka().as_u8().min(consumed[1].deaka().as_u8());
                let high = consumed[0].deaka().as_u8().max(consumed[1].deaka().as_u8());
                if pai.deaka().as_u8() < low {
                    ensure!(cans.can_chi_low, "cannot chi low");
                } else if pai.deaka().as_u8() > high {
                    ensure!(cans.can_chi_high, "cannot chi high");
                } else {
                    ensure!(cans.can_chi_mid, "cannot chi mid");
                }
            }
            Event::Pon {
                actor,
                target,
                pai,
                consumed,
            } => {
                ensure!(target != actor, "pon from itself");
                ensure!(
                    self.last_kawa_tile.filter(|&tile| tile == pai).is_some(),
                    "pon target is not the last kawa tile",
                );
                ensure!(cans.can_pon, "cannot pon");
                self.ensure_tiles_in_hand(&consumed)?;
            }

            Event::Daiminkan {
                actor,
                target,
                pai,
                consumed,
            } => {
                ensure!(target != actor, "daiminkan from itself");
                ensure!(
                    self.last_kawa_tile.filter(|&tile| tile == pai).is_some(),
                    "daiminkan target is not the last kawa tile",
                );
                ensure!(cans.can_daiminkan, "cannot daiminkan");
                self.ensure_tiles_in_hand(&consumed)?;
            }
            Event::Kakan { pai, .. } => {
                ensure!(cans.can_kakan, "cannot kakan");
                ensure!(
                    self.kakan_candidates.contains(&pai.deaka().as_u8()),
                    "cannot kakan {pai}",
                );
                self.ensure_tiles_in_hand(&[pai])?;
            }
            Event::Ankan { consumed, .. } => {
                ensure!(cans.can_ankan, "cannot ankan");
                let tile = consumed[0].deaka();
                ensure!(
                    self.ankan_candidates.contains(&tile.as_u8()),
                    "cannot ankan {tile}",
                );
                self.ensure_tiles_in_hand(&consumed)?;
            }

            Event::Hora { target, .. } => {
                if target == self.player_id {
                    ensure!(cans.can_tsumo_agari, "cannot tsumo agari");
                } else {
                    ensure!(cans.can_ron_agari, "cannot ron agari");
                }
            }

            Event::None => return Ok(()),

            _ => bail!("unexpected action {:?}", action),
        };

        Ok(())
    }

    fn ensure_tiles_in_hand(&self, tiles: &[Tile]) -> Result<()> {
        tiles.iter().try_for_each(|&tile| {
            ensure!(
                self.arrs.tehai[tile.deaka().as_usize()] > 0,
                "{tile} is not in hand",
            );
            if tile.is_aka() {
                ensure!(
                    self.akas_in_hand[tile.as_usize() - tuz!(5mr)],
                    "{tile} is not in hand",
                );
            }
            Ok(())
        })
    }
}
