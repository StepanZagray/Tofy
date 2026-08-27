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
    /// Complete API-native animation layers, retained for audit replay.
    #[serde(default)]
    pub animation: Vec<crate::p2::data::ArcFrame>,
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
    pub from: RawObservationId,
    pub action: ArcAction,
    pub to: Option<RawObservationId>,
    pub confirmed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExactExperienceGraph {
    /// Factual graph nodes never coalesce distinct API observations.
    pub nodes: BTreeMap<RawObservationId, MechanicsStateId>,
    /// Secondary mechanics lookup only; aliases do not merge factual nodes.
    #[serde(default)]
    pub mechanics_aliases: BTreeMap<MechanicsStateId, Vec<RawObservationId>>,
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
            animation: observation.animation.clone(),
        })
    }
}

impl AgentSession {
    pub fn observe(&mut self, observation: &ArcObservation) -> Result<ObservationIdentity> {
        let identity = ObservationIdentity::from_observation(observation)?;
        self.experience_graph
            .nodes
            .entry(identity.raw.clone())
            .or_insert_with(|| identity.mechanics.clone());
        let raw_ids = self
            .experience_graph
            .mechanics_aliases
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
            current_raw: current_id.raw.clone(),
            current_mechanics: current_id.mechanics.clone(),
            action: action.clone(),
            outcome: FactualActionOutcome::Confirmed {
                next_raw: next_id.raw.clone(),
                next_mechanics: next_id.mechanics.clone(),
                frame_changed: current.frame != next.frame,
                levels_delta: i32::from(next.levels_completed)
                    - i32::from(current.levels_completed),
            },
        });
        self.experience_graph.edges.push(ExperienceEdge {
            memory_index: index,
            from: current_id.raw,
            action,
            to: Some(next_id.raw),
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
            current_raw: current_id.raw.clone(),
            current_mechanics: current_id.mechanics.clone(),
            action: action.clone(),
            outcome: FactualActionOutcome::Ambiguous { mutation },
        });
        self.experience_graph.edges.push(ExperienceEdge {
            memory_index: index,
            from: current_id.raw,
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
    fn distinct_raw_observations_do_not_merge_graph_nodes() -> Result<()> {
        let first = observation("request-a");
        let second = observation("request-b");
        let mut session = AgentSession::default();
        let first_id = session.observe(&first)?;
        let second_id = session.observe(&second)?;

        assert_eq!(session.experience_graph.nodes.len(), 2);
        assert_eq!(session.experience_graph.mechanics_aliases.len(), 1);
        assert_eq!(
            session.experience_graph.mechanics_aliases[&first_id.mechanics],
            vec![first_id.raw, second_id.raw]
        );
        Ok(())
    }

    #[test]
    fn raw_identity_includes_native_dimensions_and_animation() -> Result<()> {
        let mut one_by_one = observation("request-a");
        one_by_one.animation = vec![ArcFrame::new(1, 1, vec![0])?];
        let mut two_by_one = one_by_one.clone();
        two_by_one.animation = vec![ArcFrame::new(2, 1, vec![0, 0])?];
        assert_ne!(
            ObservationIdentity::from_observation(&one_by_one)?.raw,
            ObservationIdentity::from_observation(&two_by_one)?.raw
        );

        let mut animated = one_by_one.clone();
        animated.animation = vec![ArcFrame::new(1, 1, vec![1])?, ArcFrame::new(1, 1, vec![0])?];
        assert_ne!(
            ObservationIdentity::from_observation(&one_by_one)?.raw,
            ObservationIdentity::from_observation(&animated)?.raw
        );
        Ok(())
    }

    #[test]
    fn animation_evidence_round_trips_byte_for_byte() -> Result<()> {
        let mut observation = observation("request-a");
        observation.animation = vec![
            ArcFrame::new(1, 1, vec![1])?,
            ArcFrame::new(2, 1, vec![2, 3])?,
        ];
        let mut session = AgentSession::default();
        session.observe(&observation)?;

        let restored: AgentSession = serde_json::from_slice(&serde_json::to_vec(&session)?)?;
        assert_eq!(restored.observations[0].animation, observation.animation);
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
