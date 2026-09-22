#ifndef HR_EDL_TYPES_H_
#define HR_EDL_TYPES_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "open_spiel/spiel.h"
#include "hr_edl/math.h"

namespace open_spiel {
using StatePtr = std::unique_ptr<open_spiel::State>;
}

namespace hr_edl {
using ActionAndProb = std::pair<open_spiel::Action, double>; // Action: int64_t
using Cfv = double;
using CfActionValues = std::vector<Cfv>;
// Store the counterfactual action values and expected value at a node.
struct CfValues {
  CfValues() = default;
  CfValues(CfValues&&) = default;
  CfValues(const CfValues&) = default;
  CfValues(std::pair<CfActionValues, Cfv>&& pair) : v_(std::move(pair.first)), ev_(pair.second) {}
  CfValues(CfActionValues&& v, Cfv ev) : v_(std::move(v)), ev_(ev) {}
  CfValues(const CfActionValues& v, Cfv ev) : v_(v), ev_(ev) {}
  // Create num_actions counterfactual values and an expected value, all initialized to zero.
  CfValues(size_t num_actions)
      : v_(std::vector<double>(num_actions, 0)), ev_(0) {}

  // Reset counterfactual values and the expected value.
  void Reset() {
    v_.assign(v_.size(), 0);
    ev_ = 0;
  }
  size_t Size() const { return v_.size(); }
  double operator()(const std::vector<double>& policy) const {
    return InnerProduct(v_, policy);
  }
  // Return the expected value.
  double operator()() const { return ev_; }
  // Compute regret for the supplied policy.
  double Regret(const std::vector<double>& policy) const {
    return this->operator()(policy) - ev_;
  }
  // Return the counterfactual value for action i.
  Cfv& operator[](size_t i) { return v_[i]; }
  // Return the counterfactual value for action i.
  Cfv operator[](size_t i) const { return v_[i]; }

  CfActionValues v_; // Counterfactual action values.
  Cfv ev_; // Expected value.
};

// Alias for an Abseil flat hash map of key-value pairs.
// Flat storage is useful for workloads dominated by lookups.
template <class... Args>
using UVolMap = absl::flat_hash_map<Args...>; 

// Alias for an Abseil node hash map of key-value pairs.
// Node storage keeps element addresses stable across rehashes.
template <class... Args>
using UnorderedMap = absl::node_hash_map<Args...>; 

// Map information-state strings to T using flat storage.
template <class T>
using InfoStateUvm = UVolMap<std::string, T>; 

// Map information-state strings to T using node storage.
template <class T>
using InfoStateUm = UnorderedMap<std::string, T>; 

// Vector indexed by action, storing one value of type T per action.
template <class T>
using ActionMap = std::vector<T>; 

// Vector indexed by player, storing one value of type T per player.
template <class T> 
using PlayerMap = std::vector<T>; 

// Create an ActionMap<T> with one slot per distinct action.
// NumDistinctActions() returns the state-independent action count.
  // Legal action IDs lie in {0, ..., NumDistinctActions() - 1}.
  // For example, Tic-Tac-Toe has nine actions, one per board square.
template <class T>
inline ActionMap<T> NewActionMap(const open_spiel::Game& game) {
  return ActionMap<T>(game.NumDistinctActions()); 
}

// Create a PlayerMap<T> with one slot per player.
template <class T>
inline PlayerMap<T> NewPlayerMap(const open_spiel::Game& game) {
  return PlayerMap<T>(game.NumPlayers());
}

// Create one action map for each player.
template <class T>
inline PlayerMap<ActionMap<T>> NewPlayerActionMap(
    const open_spiel::Game& game) {
  return PlayerMap<ActionMap<T>>(game.NumPlayers(), NewActionMap<T>(game));
}

using ActionIdx = size_t;
using LegalActions = std::vector<open_spiel::Action>;
using LegalActionValues = std::vector<double>;

// [(action_index, T),...]
template <class T>
using ActionsAndValues = std::vector<std::pair<open_spiel::Action, T>>; // using Action = int64_t;

// Store counterfactual values and child information-state keys for each action.
struct CfValueTreeNode {
  CfValueTreeNode(size_t num_actions)
      : cf_values_(num_actions), // Initialize values to zero.
        child_keys_(num_actions, std::vector<std::string>()),// Initialize an empty list of child keys per action.
        iwrp_(0.0) {} // Start importance-weighted reach probability at zero; accumulate over histories sharing this information set.
  
  void Reset() { cf_values_.Reset(); iwrp_ = 0.0; }
  CfValues cf_values_; // Counterfactual action values and the expected value at this node.
  std::vector<std::vector<std::string>> child_keys_; // Child information-state keys.
  // Importance-weighted reach probability.
  double iwrp_;
};
}  // namespace hr_edl

#endif  // HR_EDL_TYPES_H_
