//! Deterministic, falsification-only ordering for already-safe probe plans.

use super::graph::{ActionKey, GraphFrontier};

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeClaim {
    pub candidate_index: usize,
    pub posterior_mass: f64,
    pub satisfaction_lcb: f64,
    pub protected: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateProbe {
    pub actions: Vec<ActionKey>,
    pub safe: bool,
    pub claims: Vec<ProbeClaim>,
    pub summed_noop_probability: f64,
    pub graph_repeats: usize,
}

impl CandidateProbe {
    pub fn claim_mass(&self) -> f64 {
        self.claims
            .iter()
            .map(|claim| claim.posterior_mass * claim.satisfaction_lcb.max(0.0))
            .sum()
    }

    pub fn protected_claim_count(&self) -> usize {
        self.claims.iter().filter(|claim| claim.protected).count()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProbeChoice {
    MultiGoal(CandidateProbe),
    SingleGoal(CandidateProbe),
    GraphFrontier(GraphFrontier),
}

pub fn order_multi_goal_probes(
    probes: impl IntoIterator<Item = CandidateProbe>,
) -> Vec<CandidateProbe> {
    let mut probes = probes
        .into_iter()
        .filter(|probe| probe.safe && probe.protected_claim_count() >= 2)
        .collect::<Vec<_>>();
    probes.sort_by(compare_probe_rank);
    probes
}

pub fn choose_probe(
    probes: impl IntoIterator<Item = CandidateProbe>,
    graph_frontier: Option<GraphFrontier>,
) -> Option<ProbeChoice> {
    let probes = probes.into_iter().collect::<Vec<_>>();
    if let Some(best) = order_multi_goal_probes(probes.iter().cloned())
        .into_iter()
        .next()
    {
        return Some(ProbeChoice::MultiGoal(best));
    }
    let mut singles = probes
        .into_iter()
        .filter(|probe| probe.safe && probe.protected_claim_count() == 1)
        .collect::<Vec<_>>();
    singles.sort_by(|left, right| {
        left.actions
            .len()
            .cmp(&right.actions.len())
            .then_with(|| left.actions.cmp(&right.actions))
            .then_with(|| compare_probe_rank(left, right))
    });
    if let Some(best) = singles.into_iter().next() {
        return Some(ProbeChoice::SingleGoal(best));
    }
    graph_frontier.map(ProbeChoice::GraphFrontier)
}

fn compare_probe_rank(left: &CandidateProbe, right: &CandidateProbe) -> std::cmp::Ordering {
    right
        .claim_mass()
        .total_cmp(&left.claim_mass())
        .then_with(|| right.claims.len().cmp(&left.claims.len()))
        .then_with(|| left.actions.len().cmp(&right.actions.len()))
        .then_with(|| {
            left.summed_noop_probability
                .total_cmp(&right.summed_noop_probability)
        })
        .then_with(|| left.graph_repeats.cmp(&right.graph_repeats))
        .then_with(|| left.actions.cmp(&right.actions))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::latent_planning::graph::RawObservationId;

    fn claim(index: usize, mass: f64, lcb: f64) -> ProbeClaim {
        ProbeClaim {
            candidate_index: index,
            posterior_mass: mass,
            satisfaction_lcb: lcb,
            protected: true,
        }
    }

    fn probe(actions: &[&str], claims: Vec<ProbeClaim>) -> CandidateProbe {
        CandidateProbe {
            actions: actions.iter().map(|action| (*action).into()).collect(),
            safe: true,
            claims,
            summed_noop_probability: 0.2,
            graph_repeats: 0,
        }
    }

    #[test]
    fn multi_goal_order_is_lexicographic() {
        let mut shorter = probe(&["b"], vec![claim(0, 0.4, 0.5), claim(1, 0.4, 0.5)]);
        shorter.summed_noop_probability = 0.1;
        let longer = probe(&["a", "a"], vec![claim(0, 0.4, 0.5), claim(1, 0.4, 0.5)]);
        let ordered = order_multi_goal_probes([longer, shorter.clone()]);
        assert_eq!(ordered[0], shorter);
    }

    #[test]
    fn action_key_breaks_exact_ties_deterministically() {
        let first = probe(&["a"], vec![claim(0, 0.4, 0.5), claim(1, 0.4, 0.5)]);
        let second = probe(&["b"], vec![claim(0, 0.4, 0.5), claim(1, 0.4, 0.5)]);
        let ordered = order_multi_goal_probes([second, first.clone()]);
        assert_eq!(ordered[0], first);
    }

    #[test]
    fn fallback_chain_prefers_single_then_graph_frontier() {
        let single = probe(&["z"], vec![claim(0, 0.4, 0.5)]);
        assert!(matches!(
            choose_probe([single], None),
            Some(ProbeChoice::SingleGoal(_))
        ));
        let frontier = GraphFrontier {
            prefix: vec!["x".into()],
            node: RawObservationId::new([1; 32]),
            untried_actions: vec!["y".into()],
        };
        assert!(matches!(
            choose_probe(Vec::new(), Some(frontier)),
            Some(ProbeChoice::GraphFrontier(_))
        ));
    }
}
