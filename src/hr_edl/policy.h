#ifndef HR_EDL_POLICY_H_
#define HR_EDL_POLICY_H_

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "hr_edl/containers.h"
#include "hr_edl/decision_point.h"
#include "hr_edl/enumeration.h"
#include "hr_edl/math.h"
#include "hr_edl/types.h"

namespace hr_edl {
// Abstract class inheriting open_spiel::Policy.
// For a given state, Response and GetStatePolicy return a vector and ActionsAndProbs, respectively.
class Policy : public open_spiel::Policy {
 public:
  virtual ~Policy() = default;
  // Return the current state's behavioral policy as a vector<double>.
  virtual std::vector<double> Response(
      const open_spiel::State& state) const = 0;
  // Return the behavioral policy as OpenSpiel ActionsAndProbs: [(action_index, prob), ...].
  open_spiel::ActionsAndProbs GetStatePolicy(
      const open_spiel::State& state) const override final {
    const auto legal_actions = state.LegalActions();     // Get legal actions, a subset of {0, ..., num_distinct_actions()-1}.
    open_spiel::ActionsAndProbs actions_and_probs;
    actions_and_probs.reserve(legal_actions.size());     // Get the current state's policy as a vector<double>, with one probability per action.
    const auto policy = Response(state);
    for (int a = 0; a < legal_actions.size(); ++a) {
      actions_and_probs.emplace_back(legal_actions[a], policy[a]);
    }
    return actions_and_probs;
  }
  virtual std::vector<double> GetPolicy(
      const std::string& info_state) const {
        std::cerr << "GetPolicy unimplemented." << std::endl;
        return std::vector<double>();
      };
  // Print the policy.
  virtual void Print() const {};

};


using PolicyPtr = std::unique_ptr<Policy>;
// Return the policy table for all states in game as an unordered_map<string, ActionsAndProbs>.
// Accepts const open_spiel::Game& game.
inline std::unordered_map<std::string, open_spiel::ActionsAndProbs>
ActionsAndProbsTable(const open_spiel::Game& game, const Policy& policy) {
  std::unordered_map<std::string, open_spiel::ActionsAndProbs> map;
  // Visit every game state, obtain its information-state string, and store its policy in the map.
  // ForEachState invokes the lambda f at every game state.
  ForEachState(*(game.NewInitialState()),
               [&map, &policy](const open_spiel::State& state) {
                 const auto iss = state.InformationStateString();
                 if (!Contains(map, iss)) {
                   map[iss] = policy.GetStatePolicy(state);
                 }
               });
  return map;
}

// Return the policy table for all states under root as an unordered_map<string, ActionsAndProbs>.
// Accepts DecisionPoint& root.
inline std::unordered_map<std::string, open_spiel::ActionsAndProbs>
ActionsAndProbsTable(DecisionPoint& root, const Policy& policy) {
  std::unordered_map<std::string, open_spiel::ActionsAndProbs> map;
  ForEachState(
      root,
      [&map, &policy](const DecisionPoint& dp) {
        const auto iss = dp.InformationStateString();
        if (!Contains(map, iss)) {
          map[iss] = policy.GetStatePolicy(*dp.OpenSpielStatePtr());
        }
      },
      ALL_PLAYERS);
  return map;
}


// Policy subclass constructed from an open_spiel::Policy pointer.
// Implements Response to return the current state's action probability vector.
class ResponsePolicyAdapter : public Policy {
 public:
  ResponsePolicyAdapter(std::unique_ptr<open_spiel::Policy>&& policy)
      : policy_(std::move(policy)) {}
  // Get the current state's policy as a vector<double>, with one probability per action.
  std::vector<double> Response(
      const open_spiel::State& state) const override final {
    const auto action_prob_pairs = policy_->GetStatePolicy(state);
    std::vector<double> response(action_prob_pairs.size());
    for (int i = 0; i < action_prob_pairs.size(); ++i) {
      response[i] = action_prob_pairs[i].second;
    }
    return response;
  }

  // Return the current state's policy as ActionsAndProbs: [(action_index, prob), ...].
  open_spiel::ActionsAndProbs GetStatePolicy(
      const std::string& info_state) const override final {
    return policy_->GetStatePolicy(info_state);
  }

 private:
  std::unique_ptr<open_spiel::Policy> policy_;
};

class MapPolicy;
using MapPolicyPtr = std::unique_ptr<MapPolicy>;

// Policy subclass using an unordered map (InfoStateUvm) from information-set strings to policy vectors.
// Inherits Policy and implements Response to retrieve the current state's policy.
// Supports behavioral and sequence-form policies.
class MapPolicy : public Policy {
 public:
 // 1. Default constructor with an empty policy table.
  MapPolicy() : map_({}) {}
  // 2. Construct directly from an rvalue InfoStateUvm using move semantics and its insertion-friendly structure.
  MapPolicy(InfoStateUvm<std::vector<double>>&& map) : map_(std::move(map)) {}
  // 3. Convert an InfoStateUm to the internal map representation for efficient lookup.
  MapPolicy(const InfoStateUm<std::vector<double>>& map) {
    map_.reserve(map.size());
    for (const auto& [k, v] : map) {
      map_.emplace(k, v);
    }
  }
  // 4. Construct directly from an lvalue InfoStateUvm.
  MapPolicy(const InfoStateUvm<std::vector<double>>& map) : map_(map) {}
  // 5. Construct a sequence-form policy from a Policy and root.
  MapPolicy(const Policy& policy, DecisionPoint& root) : map_() {
    Avg(policy, root);
  }
  // 6. Construct map_policy for the specified player.
  MapPolicy(const MapPolicy& map_policy, DecisionPoint& root, int player) : map_() {
    ForEachState(
        root,
        [this, player, &root, &map_policy](const DecisionPoint& dp) {
          if (dp.PlayerToAct() == player) {
            const std::string iss = dp.InformationStateString();
            auto& policy_ref = GetOrCreate<std::string, std::vector<double>>( // Get the policy reference for iss in map_; create a uniform policy if absent.
                map_, iss,
                [&dp]() {
                  const size_t num_actions = dp.NumActions();
                  return std::vector<double>(num_actions, 1.0 / num_actions);
                });
            if (map_policy.map_.contains(iss)) {
              // Copy the policy from map_policy into map_, normalizing as needed.
              map_[iss] = map_policy.Response(*dp.OpenSpielStatePtr());
            }
          }
        },
        player);
  }

  // Return an information set's policy as vector<double>, with one probability per action.
  // Return a uniform policy if the information set is absent from map_.
  std::vector<double> Response(
      const open_spiel::State& state) const override final {
    const std::string iss = state.InformationStateString();
    // Return the stored policy if the information set exists in map_.
    if (map_.contains(iss)) {
      auto policy = map_.at(iss);
      double z = 0;
      for (const auto prob : policy) {
        z += prob;
      }
      // If z != 1, normalize safely to convert a sequence-form policy to a behavioral policy.
      if (z > 1.0 || z < 1.0) {
        SafeDivide(policy, z, true);
      }
      return policy;
    } else {
      const size_t num_actions = state.LegalActions().size();
      return std::vector<double>(num_actions, 1.0 / num_actions);
    }
  }


  MapPolicyPtr Clone() const { return std::make_unique<MapPolicy>(map_); }

  // Merge a weighted contribution from another policy into the current average policy.
  void Avg(const Policy& other, DecisionPoint& root, double weight = 1.0,
           int player = ALL_PLAYERS) {
    std::vector<double> their_reach_probs(root.NumPlayers(), 1.0);
    InfoStateUvm<std::vector<double>> their_seq_probs; // map:: string, vector<double>
    for (size_t outcome_idx = 0; outcome_idx < root.NumOutcomes(0); // Artificial root node, have only one action 0
         ++outcome_idx) {
      root.Apply(0, outcome_idx); // Apply action 0 and outcome outcome_idx
      Avg_r(their_seq_probs, their_reach_probs, other, root, player); // Recursively compute the other policy's sequence probabilities and add them to the current sequence probabilities.
      root.Undo();
    }
    // Merge the weighted sequence probabilities from their_seq_probs into map_.
    for (const auto& [iss, seq_probs] : their_seq_probs) {
      const size_t num_actions = seq_probs.size();
      auto& policy_ref = GetOrCreate<std::string, std::vector<double>>( // Get the policy reference for iss in map_; create an all-zero distribution for an unreached node if absent.
          map_, iss,
          [num_actions]() { return std::vector<double>(num_actions, 0); });
      for (size_t action_idx = 0; action_idx < num_actions; ++action_idx) {
        policy_ref[action_idx] += weight * seq_probs[action_idx];
      }
    }
  }

  // Get a policy by information-state string iss, rather than open_spiel::State.
  std::vector<double> GetPolicy(const std::string& iss) const override final {

    assert (map_.contains(iss));
      auto policy = map_.at(iss);
      double z = 0;
      for (const auto prob : policy) {
        z += prob;
      }
      // If z != 1, normalize safely to convert a sequence-form policy to a behavioral policy.
      if (z > 1.0 || z < 1.0) {
        SafeDivide(policy, z, true);
      }
      return policy;

  }
  // Get the node's unnormalized sequence-form policy directly, for average-policy computation.
  const std::vector<double>& GetRawPolicy(const std::string& iss) const {
    return map_.at(iss);
  }

  // Serialize and print the policy.
  void Print() const {
    for (const auto& [iss, policy] : map_) {
      std::cout << "InfoState: " << iss << " Policy: [";
      for (size_t a = 0; a < policy.size(); ++a) {
        std::cout << policy[a];
        if (a != policy.size() - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "]" << std::endl;
    }
  }

  void Print_behavior_strategy() const {
    for (const auto& [iss, policy] : map_) {
      std::cout << "InfoState: " << iss << " Policy: [";
      // Normalize the policy and print its behavioral form.
      std::vector<double> normalized_policy = policy;
      double z = 0;
      for (const auto prob : normalized_policy) {
        z += prob;
      }
      if (z > 1.0 || z < 1.0) {
        SafeDivide(normalized_policy, z, true);
      }
      for (size_t a = 0; a < normalized_policy.size(); ++a) {
        std::cout << normalized_policy[a];
        if (a != normalized_policy.size() - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "]" << std::endl;
    }
  }


 private:
 // Recursively compute the other policy's sequence probabilities and add them to the current sequence probabilities.
  // their_seq_probs stores the other policy's sequence probabilities.
  void Avg_r(InfoStateUvm<std::vector<double>>& their_seq_probs,
             std::vector<double>& their_reach_probs, const Policy& other,
             DecisionPoint& decision_point, int player) const {
    if (decision_point.IsTerminal()) {
      return;
    }
    const auto player_to_act = decision_point.PlayerToAct();
    const double their_reach_prob = their_reach_probs[player_to_act];
    const auto their_policy =
        other.Response(*decision_point.OpenSpielStatePtr());

    std::string iss = decision_point.InformationStateString();
    // If the player is in the specified set (any player here) and the information set is absent from their_seq_probs, add its sequence probabilities.
    if (PlayerInSet(player_to_act, player) && !their_seq_probs.contains(iss)) {
      std::vector<double> seq_probs;
      seq_probs.reserve(their_policy.size());
      for (size_t action_idx = 0; action_idx < their_policy.size();
           ++action_idx) {
        seq_probs.push_back(their_reach_prob * their_policy[action_idx]);
      }
      their_seq_probs.emplace(std::move(iss), std::move(seq_probs));
    }
    // Visit all actions and outcomes, recursively computing the other policy's sequence probabilities.
    for (size_t action_idx = 0; action_idx < their_policy.size();
         ++action_idx) {
      for (size_t outcome_idx = 0;
           outcome_idx < decision_point.NumOutcomes(action_idx);
           ++outcome_idx) {
        their_reach_probs[player_to_act] =
            their_reach_prob * their_policy[action_idx];
        decision_point.Apply(action_idx, outcome_idx);
        Avg_r(their_seq_probs, their_reach_probs, other, decision_point,
              player);
        decision_point.Undo();
      }
    }
    // Update the current player's reach probability for the next node.
    their_reach_probs[player_to_act] = their_reach_prob;
  }

  // Zero the policy table.
  void Clear() {
    for (auto& [iss, policy] : map_) {
      for (auto& prob : policy) {
        prob = 0.0;
      }
    }
  }

 private:
 // Map information-set strings iss to policy vectors.
  mutable InfoStateUvm<std::vector<double>> map_; // Private mutable storage, modifiable internally but not directly by callers.
};

// Return a MapPolicy with an empty map {}, representing a uniform random policy.
inline MapPolicy UniformRandomPolicy() { return MapPolicy(); }

class PolicyRefProfile;
// Abstract player-mapped policy class inheriting Policy.
// Stores the player mapping in std::vector<size_t>.
// Manages policies for different players in multiplayer games.
// Access each player's policy by index.
class PlayerMapProfile : public Policy {
 public:
 // The default player mapping follows the game tree's default player order.
  PlayerMapProfile(size_t num_players) : player_map_(Range(num_players)) {}
  // Accept an rvalue reference specifying the player-order mapping.
  PlayerMapProfile(std::vector<size_t>&& player_map)
      : player_map_(std::move(player_map)) {}
  // Accept a const reference specifying the player-order mapping.
  PlayerMapProfile(const std::vector<size_t>& player_map)
      : player_map_(player_map) {}

  // Return this state's player policy under player_map_ as vector<double>.
  std::vector<double> Response(
      const open_spiel::State& state) const override final {
    return (*this)[state.CurrentPlayer()]->Response(state);
  }
  // Indexing interface: [player] returns the referenced player policy.
  virtual const Policy* operator[](size_t player) const = 0;

 protected:
  size_t NumPlayers() const { return player_map_.size(); }
  virtual std::vector<const Policy*> Policies() const = 0;

 protected:
  std::vector<size_t> player_map_; // Player mapping.
};


// Player-mapped policy class inheriting PlayerMapProfile.
// Stores player policy pointers in std::vector<const Policy*>.
// Manages policies for different players in multiplayer games.
// Access each player's policy by index.
// References policies without owning them.
class PolicyRefProfile : public PlayerMapProfile {
 public:
 // 1. Rvalue policies with default player order.
  PolicyRefProfile(std::vector<const Policy*>&& policies)
      : PlayerMapProfile(policies.size()), policies_(std::move(policies)) {}
  // 2. Rvalue policies with specified player order.
  PolicyRefProfile(std::vector<const Policy*>&& policies,
                   std::vector<size_t>&& player_map)
      : PlayerMapProfile(std::move(player_map)),
        policies_(std::move(policies)) {}
  // 3. Const-reference policies with default player order.
  PolicyRefProfile(const std::vector<const Policy*>& policies)
      : PlayerMapProfile(policies.size()), policies_(policies) {}
  // 4. Const-reference policies with specified player order.
  PolicyRefProfile(const std::vector<const Policy*>& policies,
                   const std::vector<size_t>& player_map)
      : PlayerMapProfile(player_map), policies_(policies) {}
  template <class PolicyLike>
  // 5. Construct PolicyRefProfile from PolicyLike pointers with default player order.
  PolicyRefProfile(const std::vector<std::unique_ptr<PolicyLike>>& policies)
      : PlayerMapProfile(policies.size()) {
    policies_.reserve(policies.size());
    for (size_t i = 0; i < policies.size(); ++i) {
      policies_.push_back(static_cast<const Policy*>(policies[i].get()));
    }
  }
  template <class PolicyLike>
  // 6. Construct from PolicyLike pointers with default player order, replacing the specified player's policy with learner.
  PolicyRefProfile(const Policy* learner,
                   const std::vector<std::unique_ptr<PolicyLike>>& compatriots,
                   size_t learner_idx)
      : PlayerMapProfile(compatriots.size()) {
    policies_.reserve(compatriots.size());
    for (size_t player = 0; player < compatriots.size(); ++player) {
      policies_.push_back(
          (player == learner_idx)
              ? learner
              : static_cast<const Policy*>(compatriots[player].get()));
    }
  }

  // Replace the policy at learner_idx in player_map with learner and return a new PolicyRefProfile.
  PolicyRefProfile WithSubstitute(const Policy* learner,
                                  size_t learner_idx) const {
    std::vector<const Policy*> policies = Policies();
    auto player_map = player_map_;
    player_map[learner_idx] = policies.size();
    policies.push_back(learner);
    return PolicyRefProfile(std::move(policies), std::move(player_map));
  }

  // Indexing operator [player] returns the referenced player policy.
  const Policy* operator[](size_t player) const override {
    return policies_[player_map_[player]];
  }

 protected:
  std::vector<const Policy*> Policies() const override { return policies_; }

 private:
 // Nonowning list of player Policy* pointers.
  std::vector<const Policy*> policies_;
};

// PolicyProfile is a policy-profile class inheriting PlayerMapProfile.
// Stores player policy pointers in std::vector<PolicyPtr>.
// Manages policies for different players in multiplayer games.
// Access player policies by index and replace a player's policy with WithSubstitute.
// Owns policies through PolicyPtr smart pointers.
class PolicyProfile : public PlayerMapProfile {
 public:
 // 1. Rvalue policy-pointer vector with default player positions.
  PolicyProfile(std::vector<PolicyPtr>&& policies)
      : PlayerMapProfile(policies.size()), policies_(std::move(policies)) {}
  // 2. Rvalue PolicyProfile with default player positions.
  PolicyProfile(PolicyProfile&& profile)
      : PolicyProfile(std::move(profile.policies_)) {} // Take profile's policy-pointer vector and delegate to the first constructor.
  template <class PolicyLike>
  // 3. Construct PolicyProfile from PolicyLike pointers, e.g. learners, with default player order.
  PolicyProfile(std::vector<std::unique_ptr<PolicyLike>>&& policies)
      : PlayerMapProfile(policies.size()) {
    policies_.reserve(policies.size());
    for (size_t i = 0; i < policies.size(); ++i) {
      policies_.push_back(PolicyPtr(std::move(policies[i])));
    }
  }
  // Construct PolicyProfile from MapPolicy with default player order.
  PolicyProfile(MapPolicy& map_policy, DecisionPoint& root)
      : PlayerMapProfile(root.NumPlayers()) {
    policies_.reserve(root.NumPlayers());
    for (size_t player = 0; player < root.NumPlayers(); ++player) {
      policies_.push_back(
          PolicyPtr(new MapPolicy(map_policy, root, player)));
    }
  }

  // Replace the policy for learner_idx and return a new PolicyRefProfile.
  // Map that player's index to a new entry appended to policies for learner.
  PolicyRefProfile WithSubstitute(const Policy* learner,
                                  size_t learner_idx) const {
    std::vector<const Policy*> policies = Policies(); // Get a copy of the vector of all player policy pointers.
    auto player_map = player_map_; // Copy the player mapping [0, 1, 2, ...].
    player_map[learner_idx] = policies.size(); // Place learner in the newly appended policy slot.
    policies.push_back(learner); // Append learner to the policy vector as the replacement policy.
    return PolicyRefProfile(std::move(policies), std::move(player_map)); // Return a new PolicyRefProfile.
  }
// Operator overload.
  const Policy* operator[](size_t player) const override {
    return policies_[player_map_[player]].get();
  }

  // Print the policy.
  void Print() const {
    for (size_t player = 0; player < policies_.size(); ++player) {
      std::cout << "Player " << player << " Policy:" << std::endl;
      policies_[player]->Print();
    }
  }

 protected:
 // Return a copy of the vector of all player policy pointers.
  std::vector<const Policy*> Policies() const override {
    std::vector<const Policy*> policies;
    policies.reserve(policies_.size());
    for (size_t i = 0; i < policies_.size(); ++i) {
      policies.push_back(policies_[i].get());// Raw pointer managed by the smart pointer.
    }
    return policies;
  }

 private:
  std::vector<PolicyPtr> policies_; // Owning PolicyPtr smart pointers.
};

}  // namespace hr_edl

#endif  // HR_EDL_POLICY_H_
