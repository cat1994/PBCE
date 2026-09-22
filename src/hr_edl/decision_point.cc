#include "hr_edl/decision_point.h"

#include "absl/container/flat_hash_set.h"
#include "hr_edl/containers.h"
#include "hr_edl/samplers.h"
namespace hr_edl {

double DecisionPoint::ApplySampledOutcome(size_t action, double random_number) {
  const std::vector<double>& pi = OutcomeProbabilities(action);
  const size_t outcome = SampleActionIndex(pi, random_number);
  const double prob = pi[outcome];
  Apply(action, outcome);
  return prob;
}

// Construct from an rvalue OpenSpiel state pointer.
CachedDecisionPoint::CachedDecisionPoint(open_spiel::StatePtr&& root, // Root OpenSpiel state pointer.
                                         bool save_root, bool save_terminals) // false, true
    : DecisionPoint(root->NumPlayers(), root->NumDistinctActions()), // Initialize the base class with the player and action counts.
      idx_(0),
      histories_(), // Initialize the decision-history cache.
      save_terminals_(save_terminals) {
  // Cache a terminal root, retaining its state only when requested.
  if (root->IsTerminal()) {
    if (save_terminals_) {
      histories_.emplace_back(std::move(root), idx_);
    } else {
      histories_.emplace_back(root->Returns(), idx_);
    }
  // Preserve a chance root only when save_root is enabled.
  } else if (root->IsChanceNode() && save_root) { // save_root: false
    histories_.emplace_back(std::move(root));
    histories_[idx_].outcomes_.emplace_back();
    RecursiveCache(*(histories_[idx_].os_state_), 1.0, 0);
    // Otherwise represent the input state below an artificial root.
  } else {
    // Create the artificial root through the default HistoryCache constructor.
    histories_.emplace_back(); // The artificial root occupies index zero.
    histories_[idx_].outcomes_.emplace_back(); // Create the artificial root's single action outcome list.
    RecursiveCache(std::move(root), 1.0, 0); // Cache all non-chance outcomes below artificial-root action zero.
  }
}

// Expand a chance node into cached non-chance outcomes.
// child: the chance state to expand.
void CachedDecisionPoint::RecursiveCache(const open_spiel::State& child,
                                         double prob, size_t aidx) {
  // Visit every chance outcome.
  for (const auto [outcome, next_prob] : child.ChanceOutcomes()) {
    RecursiveCache(child.Child(outcome), prob * next_prob, aidx); // Collapse consecutive chance nodes by multiplying their probabilities.
  }
}

// Recursively cache non-chance descendants of a decision point.
// Represent intervening chance nodes implicitly through outcome probabilities.
// child: child state pointer.
// prob: accumulated chance probability from the parent decision point.
void CachedDecisionPoint::RecursiveCache(open_spiel::StatePtr&& child,
                                         double prob, size_t aidx) {
  // Collapse consecutive chance nodes and recursively expand their outcomes.
  if (child->IsChanceNode()) {
    RecursiveCache(*child, prob, aidx); // Expand the referenced state with the accumulated chance probability.
    return;
  } else if (!child->IsTerminal()) { // A nonterminal decision node.
    // Skip forced actions to avoid redundant decision nodes.
    const auto actions = child->LegalActions();
    if (actions.size() < 2) {
      RecursiveCache(child->Child(actions[0]), prob, aidx);
      return; // The forced-action descendant has already been cached.
    }
  } else if (!save_terminals_) { // Retain only the returns of terminal states when requested.
    histories_[idx_].outcomes_[aidx].PushBack(histories_.size(), prob); // Append the child index and chance probability.
    histories_.emplace_back(child->Returns(), idx_); // Store terminal returns without retaining the state.
    return;
  }
  // Non-trivial decision node or terminal to be saved.
  // Append the non-chance decision or retained terminal node to the current outcomes.
  // The new child index is the current history-cache size.
  histories_[idx_].outcomes_[aidx].PushBack(histories_.size(), prob);
  // Append the state to the history cache.
  histories_.emplace_back(std::move(child), idx_);
}

void CachedDecisionPoint::CacheOutcomes() {
  // Terminal nodes and previously expanded nodes need no further caching.
  if (IsTerminal() || histories_[idx_].outcomes_.size() > 0) {
    return;
  }
  // Query legal actions; occupied cells, for example, are unavailable in tic-tac-toe.
  const auto actions = histories_[idx_].os_state_->LegalActions();
  // Cache the successor outcomes of each legal action.
  for (size_t aidx = 0; aidx < actions.size(); ++aidx) {
    histories_[idx_].outcomes_.emplace_back(); // Create an outcome list for this action.
    // Cache successor histories; Kuhn information states encode a private card and betting sequence.
    RecursiveCache(histories_[idx_].os_state_->Child(actions[aidx]), 1.0, aidx);
  }
}

// Recursively visit reachable information states and invoke f once for each.
void _ForEachState(absl::flat_hash_set<std::string>& already_observed,
                   DecisionPoint& decision_point,
                   const std::function<void(const DecisionPoint&)>& f,
                   int player) {
  if (decision_point.IsTerminal()) {
    return;
  }
  const std::string info_state = decision_point.InformationStateString();
  // Select the requested player, or every player when player is negative.
  // Invoke f only for previously unobserved information-state keys.
  if (player < 0 || player == decision_point.PlayerToAct()) {
    if (!already_observed.contains(info_state)) {
      f(decision_point);
      already_observed.insert(std::move(info_state));
    };
  }
  // Visit every action and successor outcome.
  for (size_t a = 0; a < decision_point.NumActions(); ++a) {
    for (size_t outcome = 0; outcome < decision_point.NumOutcomes(a);
         ++outcome) {
      decision_point.Apply(a, outcome);
      _ForEachState(already_observed, decision_point, f, player);
      decision_point.Undo();
    }
  }
}

// Visit reachable information states from the root and invoke f once for each.
void ForEachState(DecisionPoint& root,
                  const std::function<void(const DecisionPoint&)>& f,
                  int player) {
  absl::flat_hash_set<std::string> already_observed;
  if (root.NumActions() < 2) { // Traverse the single artificial-root action directly.
    for (size_t outcome = 0; outcome < root.NumOutcomes(0); ++outcome) { // A single action can lead to several successor histories through chance.
      root.Apply(0, outcome); // Apply action zero and the selected outcome.
      _ForEachState(already_observed, root, f, player);
      root.Undo();
    }
  } else {
    _ForEachState(already_observed, root, f, player);
  }
}

// Count reachable information states for the selected player.
int NumStates(DecisionPoint& root, int player) {
  int count = 0;
  ForEachState(
      root, [&count](const DecisionPoint& _) { ++count; }, player);
  return count;
}

// Count reachable information states in which each action is legal.
ActionMap<int> NumStatesWithAction(DecisionPoint& root, int player) {
  ActionMap<int> counts(root.NumDistinctActions(), 0);
  ForEachState(
      root,
      [&counts](const DecisionPoint& successor) {
        for (const open_spiel::Action a :
             successor.OpenSpielStatePtr()->LegalActions()) {
          ++counts[a];
        }
      },
      player);
  return counts;
}

}  // namespace hr_edl
