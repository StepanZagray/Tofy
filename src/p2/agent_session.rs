//! Factual identity, memory, and exact experience graph for live ARC sessions.
//!
//! This first native-agent slice deliberately contains no hypothesis scoring or
//! planning. It records only what the environment actually returned.

use crate::p2::arc3_live::{AmbiguousMutation, ArcObservation};
use crate::p2::data::ArcAction;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RawObservationId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MechanicsStateId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationIdentity {
    pub raw: RawObservationId,
    pub mechanics: MechanicsStateId,
    pub game_id: String,
    pub guid: String,
    pub state: String,
    pub levels_completed: u16,
    pub win_levels: u16,
    pub available_actions: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum FactualActionOutcome {
    Confirmed {
        next_raw: RawObservationId,
        next_mechanics: MechanicsStateId,
        frame_changed: bool,
        levels_delta: i32,
    },
    Ambiguous {
        mutation: AmbiguousMutation,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactualMemoryEntry {
    pub index: usize,
    pub current_raw: RawObservationId,
    pub current_mechanics: MechanicsStateId,
    pub action: ArcAction,
    pub outcome: FactualActionOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceEdge {
    pub memory_index: usize,
    pub from: MechanicsStateId,
    pub action: ArcAction,
    pub to: Option<MechanicsStateId>,
    pub confirmed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExactExperienceGraph {
    pub nodes: BTreeMap<MechanicsStateId, Vec<RawObservationId>>,
    pub edges: Vec<ExperienceEdge>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentSession {
    pub observations: Vec<ObservationIdentity>,
    pub factual_memory: Vec<FactualMemoryEntry>,
    pub experience_graph: ExactExperienceGraph,
}

fn digest<T: Serialize>(domain: &[u8], value: &T) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(serde_json::to_vec(value)?);
    Ok(format!("sha256:{:x}", hash.finalize()))
}

impl ObservationIdentity {
    pub fn from_observation(observation: &ArcObservation) -> Result<Self> {
        let raw = RawObservationId(digest(b"tofy.raw-observation.v1", observation)?);
        let mechanics_payload = (
            &observation.game_id,
            &observation.frame,
            &observation.state,
            observation.levels_completed,
            observation.win_levels,
            &observation.available_actions,
        );
        let mechanics = MechanicsStateId(digest(b"tofy.mechanics-state.v1", &mechanics_payload)?);
        Ok(Self {
            raw,
            mechanics,
            game_id: observation.game_id.clone(),
            guid: observation.guid.clone(),
            state: observation.state.clone(),
            levels_completed: observation.levels_completed,
            win_levels: observation.win_levels,
            available_actions: observation.available_actions.clone(),
        })
    }
}

impl AgentSession {
    pub fn observe(&mut self, observation: &ArcObservation) -> Result<ObservationIdentity> {
        let identity = ObservationIdentity::from_observation(observation)?;
        let raw_ids = self
            .experience_graph
            .nodes
            .entry(identity.mechanics.clone())
            .or_default();
        if !raw_ids.contains(&identity.raw) {
            raw_ids.push(identity.raw.clone());
        }
        if !self
            .observations
            .iter()
            .any(|seen| seen.raw == identity.raw)
        {
            self.observations.push(identity.clone());
        }
        Ok(identity)
    }

    pub fn record_confirmed(
        &mut self,
        current: &ArcObservation,
        action: ArcAction,
        next: &ArcObservation,
    ) -> Result<()> {
        let current_id = self.observe(current)?;
        let next_id = self.observe(next)?;
        let index = self.factual_memory.len();
        self.factual_memory.push(FactualMemoryEntry {
            index,
            current_raw: current_id.raw,
            current_mechanics: current_id.mechanics.clone(),
            action: action.clone(),
            outcome: FactualActionOutcome::Confirmed {
                next_raw: next_id.raw,
                next_mechanics: next_id.mechanics.clone(),
                frame_changed: current.frame != next.frame,
                levels_delta: i32::from(next.levels_completed)
                    - i32::from(current.levels_completed),
            },
        });
        self.experience_graph.edges.push(ExperienceEdge {
            memory_index: index,
            from: current_id.mechanics,
            action,
            to: Some(next_id.mechanics),
            confirmed: true,
        });
        Ok(())
    }

    pub fn record_ambiguous(
        &mut self,
        current: &ArcObservation,
        action: ArcAction,
        mutation: AmbiguousMutation,
    ) -> Result<()> {
        let current_id = self.observe(current)?;
        let index = self.factual_memory.len();
        self.factual_memory.push(FactualMemoryEntry {
            index,
            current_raw: current_id.raw,
            current_mechanics: current_id.mechanics.clone(),
            action: action.clone(),
            outcome: FactualActionOutcome::Ambiguous { mutation },
        });
        self.experience_graph.edges.push(ExperienceEdge {
            memory_index: index,
            from: current_id.mechanics,
            action,
            to: None,
            confirmed: false,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::data::ArcFrame;

    fn observation(guid: &str) -> ArcObservation {
        ArcObservation {
            game_id: "game".into(),
            guid: guid.into(),
            frame: ArcFrame::new(64, 64, vec![0; 64 * 64]).unwrap(),
            animation: Vec::new(),
            full_reset: false,
            state: "NOT_FINISHED".into(),
            levels_completed: 0,
            win_levels: 1,
            available_actions: vec![1, 2],
        }
    }

    #[test]
    fn raw_and_mechanics_identity_are_deliberately_distinct() -> Result<()> {
        let first = ObservationIdentity::from_observation(&observation("request-a"))?;
        let second = ObservationIdentity::from_observation(&observation("request-b"))?;
        assert_ne!(first.raw, second.raw);
        assert_eq!(first.mechanics, second.mechanics);
        Ok(())
    }

    #[test]
    fn ambiguous_action_has_no_fabricated_graph_destination() -> Result<()> {
        let current = observation("request-a");
        let action = ArcAction::new(1, None, None)?;
        let mutation = AmbiguousMutation {
            operation: "ACTION".into(),
            game_id: Some("game".into()),
            guid: Some("request-a".into()),
            action: Some(action.clone()),
            cause: "timeout".into(),
        };
        let mut session = AgentSession::default();
        session.record_ambiguous(&current, action, mutation)?;
        assert_eq!(session.factual_memory.len(), 1);
        assert!(!session.experience_graph.edges[0].confirmed);
        assert!(session.experience_graph.edges[0].to.is_none());
        Ok(())
    }
}
