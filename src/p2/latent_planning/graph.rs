//! Append-only exact observations and factual transitions for Phase A.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RawObservationId([u8; 32]);

impl RawObservationId {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ActionKey(pub String);

impl From<&str> for ActionKey {
    fn from(value: &str) -> Self {
        Self(value.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminalChannel {
    Satisfied,
    Failed,
    Exhausted,
    Ordinary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPredictionRecord {
    pub decoded_gameplay_frame: Option<Vec<u8>>,
    pub trusted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactualEdge {
    pub action: ActionKey,
    pub next_raw_id: Option<RawObservationId>,
    pub board_effect: Vec<u8>,
    pub terminal: TerminalChannel,
    pub action_cost: u32,
    pub model_prediction: ModelPredictionRecord,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedStateNode {
    pub exact_observation: Vec<u8>,
    pub legal_actions: BTreeSet<ActionKey>,
    pub tried_actions: BTreeSet<ActionKey>,
    pub edges: BTreeMap<ActionKey, FactualEdge>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphFrontier {
    pub prefix: Vec<ActionKey>,
    pub node: RawObservationId,
    pub untried_actions: Vec<ActionKey>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservedGraphError {
    ObservationCollision(RawObservationId),
    UnknownNode(RawObservationId),
    IllegalAction(ActionKey),
    EdgeAlreadyRecorded(ActionKey),
}

impl fmt::Display for ObservedGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ObservationCollision(id) => write!(f, "raw observation collision for {id:?}"),
            Self::UnknownNode(id) => write!(f, "unknown raw observation {id:?}"),
            Self::IllegalAction(action) => write!(f, "action {:?} was not advertised", action.0),
            Self::EdgeAlreadyRecorded(action) => {
                write!(f, "action {:?} already has an edge", action.0)
            }
        }
    }
}

impl std::error::Error for ObservedGraphError {}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedStateGraph {
    nodes: BTreeMap<RawObservationId, ObservedStateNode>,
}

impl ObservedStateGraph {
    pub fn insert_node(
        &mut self,
        id: RawObservationId,
        exact_observation: Vec<u8>,
        legal_actions: impl IntoIterator<Item = ActionKey>,
    ) -> Result<(), ObservedGraphError> {
        let node = ObservedStateNode {
            exact_observation,
            legal_actions: legal_actions.into_iter().collect(),
            tried_actions: BTreeSet::new(),
            edges: BTreeMap::new(),
        };
        match self.nodes.get(&id) {
            Some(existing) if existing == &node => Ok(()),
            Some(_) => Err(ObservedGraphError::ObservationCollision(id)),
            None => {
                self.nodes.insert(id, node);
                Ok(())
            }
        }
    }

    pub fn node(&self, id: RawObservationId) -> Option<&ObservedStateNode> {
        self.nodes.get(&id)
    }

    pub fn append_edge(
        &mut self,
        from: RawObservationId,
        edge: FactualEdge,
    ) -> Result<(), ObservedGraphError> {
        let node = self
            .nodes
            .get_mut(&from)
            .ok_or(ObservedGraphError::UnknownNode(from))?;
        if !node.legal_actions.contains(&edge.action) {
            return Err(ObservedGraphError::IllegalAction(edge.action));
        }
        if node.edges.contains_key(&edge.action) {
            return Err(ObservedGraphError::EdgeAlreadyRecorded(edge.action));
        }
        node.tried_actions.insert(edge.action.clone());
        node.edges.insert(edge.action.clone(), edge);
        Ok(())
    }

    /// Retrodiction may only revoke model trust. Factual edges stay intact.
    pub fn mark_prediction_untrusted(
        &mut self,
        from: RawObservationId,
        action: &ActionKey,
    ) -> Result<(), ObservedGraphError> {
        let node = self
            .nodes
            .get_mut(&from)
            .ok_or(ObservedGraphError::UnknownNode(from))?;
        let edge = node
            .edges
            .get_mut(action)
            .ok_or_else(|| ObservedGraphError::IllegalAction(action.clone()))?;
        edge.model_prediction.trusted = false;
        Ok(())
    }

    pub fn retrodict(
        &mut self,
        from: RawObservationId,
        action: &ActionKey,
        exact_match: bool,
    ) -> Result<(), ObservedGraphError> {
        if !exact_match {
            self.mark_prediction_untrusted(from, action)?;
        }
        Ok(())
    }

    pub fn nearest_untried_frontier(&self, start: RawObservationId) -> Option<GraphFrontier> {
        let start_node = self.nodes.get(&start)?;
        if let Some(untried_actions) = untried_actions(start_node) {
            return Some(GraphFrontier {
                prefix: Vec::new(),
                node: start,
                untried_actions,
            });
        }

        let mut visited = BTreeSet::from([start]);
        let mut queue = VecDeque::from([(start, Vec::new())]);
        while let Some((node_id, prefix)) = queue.pop_front() {
            let node = self.nodes.get(&node_id)?;
            for (action, edge) in &node.edges {
                let Some(next) = edge.next_raw_id else {
                    continue;
                };
                if !visited.insert(next) {
                    continue;
                }
                let mut next_prefix = prefix.clone();
                next_prefix.push(action.clone());
                let next_node = self.nodes.get(&next)?;
                if let Some(untried_actions) = untried_actions(next_node) {
                    return Some(GraphFrontier {
                        prefix: next_prefix,
                        node: next,
                        untried_actions,
                    });
                }
                queue.push_back((next, next_prefix));
            }
        }
        None
    }
}

fn untried_actions(node: &ObservedStateNode) -> Option<Vec<ActionKey>> {
    let actions = node
        .legal_actions
        .difference(&node.tried_actions)
        .cloned()
        .collect::<Vec<_>>();
    (!actions.is_empty()).then_some(actions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(byte: u8) -> RawObservationId {
        RawObservationId::new([byte; 32])
    }

    fn edge(action: &str, next: Option<RawObservationId>) -> FactualEdge {
        FactualEdge {
            action: action.into(),
            next_raw_id: next,
            board_effect: vec![1],
            terminal: TerminalChannel::Ordinary,
            action_cost: 1,
            model_prediction: ModelPredictionRecord {
                decoded_gameplay_frame: Some(vec![1]),
                trusted: true,
            },
        }
    }

    #[test]
    fn factual_nodes_and_edges_are_append_only() {
        let mut graph = ObservedStateGraph::default();
        graph.insert_node(id(1), vec![1], ["a".into()]).unwrap();
        graph.append_edge(id(1), edge("a", None)).unwrap();
        assert!(matches!(
            graph.append_edge(id(1), edge("a", None)),
            Err(ObservedGraphError::EdgeAlreadyRecorded(_))
        ));
        assert!(matches!(
            graph.insert_node(id(1), vec![2], ["a".into()]),
            Err(ObservedGraphError::ObservationCollision(_))
        ));
    }

    #[test]
    fn retrodiction_mismatch_only_revokes_prediction_trust() {
        let mut graph = ObservedStateGraph::default();
        graph.insert_node(id(1), vec![1], ["a".into()]).unwrap();
        graph.append_edge(id(1), edge("a", None)).unwrap();
        graph.retrodict(id(1), &"a".into(), false).unwrap();
        let edge = &graph.node(id(1)).unwrap().edges[&ActionKey::from("a")];
        assert!(!edge.model_prediction.trusted);
        assert_eq!(edge.board_effect, vec![1]);
    }

    #[test]
    fn frontier_bfs_returns_shortest_prefix_with_action_order_ties() {
        let mut graph = ObservedStateGraph::default();
        graph
            .insert_node(id(1), vec![1], ["a".into(), "b".into()])
            .unwrap();
        graph.insert_node(id(2), vec![2], ["c".into()]).unwrap();
        graph.insert_node(id(3), vec![3], ["d".into()]).unwrap();
        graph.insert_node(id(4), vec![4], ["e".into()]).unwrap();
        graph.append_edge(id(1), edge("a", Some(id(3)))).unwrap();
        graph.append_edge(id(1), edge("b", Some(id(2)))).unwrap();
        graph.append_edge(id(2), edge("c", Some(id(4)))).unwrap();
        let frontier = graph.nearest_untried_frontier(id(1)).unwrap();
        assert_eq!(frontier.prefix, vec![ActionKey::from("a")]);
        assert_eq!(frontier.node, id(3));
    }

    #[test]
    fn frontier_never_offers_a_tried_action() {
        let mut graph = ObservedStateGraph::default();
        graph
            .insert_node(id(1), vec![1], ["a".into(), "b".into()])
            .unwrap();
        graph.append_edge(id(1), edge("a", None)).unwrap();
        let frontier = graph.nearest_untried_frontier(id(1)).unwrap();
        assert_eq!(frontier.untried_actions, vec![ActionKey::from("b")]);
    }
}
