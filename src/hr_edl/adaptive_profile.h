#ifndef HR_EDL_ADAPTIVE_PROFILE_H_
#define HR_EDL_ADAPTIVE_PROFILE_H_

#include "hr_edl/best_response.h"
#include "hr_edl/decision_point.h"
#include "hr_edl/policy_evaluation.h"
#include "hr_edl/samplers.h"
#include "hr_edl/tabular_learner.h"
#include "hr_edl/types.h"

namespace hr_edl {
class AdaptiveProfile;
using AdaptiveProfilePtr =
    std::unique_ptr<AdaptiveProfile>;  // Owning pointer to an AdaptiveProfile instance or subclass.

// Abstract interface for adaptive algorithm profiles.
class AdaptiveProfile {
 public:
  virtual ~AdaptiveProfile() = default;
  // Return the specified player's policy pointer.
  virtual Policy* Strategy(size_t player) const = 0;
  // Return the average counterfactual value against each opponent.
  Cfv Ev(DecisionPoint& root, MccfrSampler& sampler, const PolicyProfile& compatriots) const {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto [v, _] =
          PolicyValue(root, player, compatriots.WithSubstitute(Strategy(player), player), sampler);
      avg += (v - avg) / (player + 1.0);  // Update the mean.
    }
    return avg;
  }
  // Update the policy and return the expected value.
  virtual Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
                                const PolicyProfile& compatriots) = 0;
  // Update players in turn and return the expected value.
  virtual Cfv UpdateAlternateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler) = 0;

  virtual PolicyProfile Frozen() const = 0;
  MapPolicy ToMapPolicy(DecisionPoint& root) const { return MapPolicy(Frozen(), root); }

  // Traverse the game tree and compute total/maximum phi regrets using the average policy.
  // virtual std::tuple<double, double, double, double> ReachedPhiRegret(DecisionPoint& root,
  //                                                                     MccfrSampler& sampler) {
  //   return {0.0, 0.0, 0.0, 0.0};
  // }
  virtual std::tuple<double, double, double, double> PhiRegrets() const {
    return {0.0, 0.0, 0.0, 0.0};
  }
  virtual void ShrinkEpsilon(double new_epsilon) {}
  virtual double Epsilon() const { return 0.0; }
  virtual StageRegretMetrics StageMetrics(double w_numerical_eps) const { return {}; }
  virtual void ResetStageEvaluation() {}
  virtual void ResetSolverRegrets() {}
  virtual void SetCfrPlus(bool enabled) {}
  virtual double SolverRegretL1() const { return 0.0; }
  virtual double StageOriginalRegretL1() const { return 0.0; }
  virtual double StageExposureSum() const { return 0.0; }
};

// CFR algorithm profile inheriting AdaptiveProfile.
class CfTreeLearnerProfile : public AdaptiveProfile {
 public:
  // learners_: vector of owning CfValueTreeLearnerPtr pointers, one per player.
  // evaluators_: one PolicyCfValueTreeEvaluator per player, computing that player's CFV tree
  // from the root, player policies, and sampler.
  CfTreeLearnerProfile(std::vector<CfValueTreeLearnerPtr>&& learners)
      : learners_(std::move(learners)), evaluators_(NewEvaluators()) {}

  Policy* Strategy(size_t player) const override final { return learners_[player].get(); }
  // Update the policy and return the expected value.
  // compatriots: policy profile with the policy at player replaced by the current player's policy.
  Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
                        const PolicyProfile& compatriots) override final {
    double avg = 0.0;
    // Compute the current player's CFV tree at each player position for the subsequent policy update.
    for (int player = 0; player < root.NumPlayers(); ++player) {
      // Get the initial sibling information-state strings and CFV tree to update.
      const auto [v, initial_info_states, cf_value_tree, _] =
          evaluators_[player].ComputeCfValueTreeEvaluation(
              root, compatriots.WithSubstitute(Strategy(player), player), sampler);
      // Update the player's policy using the initial information-state strings and CFV tree.
      learners_[player]->Update(initial_info_states, *cf_value_tree);
      // Update the mean over all players; equivalent to avg = (avg*player + v)/(player+1).
      avg += (v - avg) / (player + 1.0);
    }
    return avg;
  }
  // Update players in turn and return the expected value.
  Cfv UpdateAlternateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto [v, initial_info_states, cf_value_tree, _] =
          evaluators_[player].ComputeCfValueTreeEvaluation(root, PolicyRefProfile(learners_),
                                                           sampler);
      learners_[player]->Update(initial_info_states, *cf_value_tree);
      avg += (v - avg) / (player + 1.0);
    }
    return avg;
  }
  // Equivalent to return PolicyProfile(Clone(learners_)), implicitly invoking the return type's constructor.
  PolicyProfile Frozen() const override final { return Clone(learners_); }

 private:
  // Create one PolicyCfValueTreeEvaluator per player to record that player's CFV tree.
  std::vector<PolicyCfValueTreeEvaluator> NewEvaluators() {
    std::vector<PolicyCfValueTreeEvaluator> l;
    for (size_t i = 0; i < learners_.size(); ++i) {
      l.emplace_back(i);  // = l.push_back(PolicyCfValueTreeEvaluator(i));
    }
    return l;
  }

 private:
  std::vector<CfValueTreeLearnerPtr> learners_;
  std::vector<PolicyCfValueTreeEvaluator> evaluators_;
};

// // Reward-transformation CFR profile inheriting AdaptiveProfile.
// class RTCfTreeLearnerProfile : public AdaptiveProfile {
//  public:
//   RTCfTreeLearnerProfile(std::vector<CfValueTreeLearnerPtr>&& learners, double rt_weight, size_t
//   T)
//       : learners_(std::move(learners)),
//         evaluators_(NewEvaluators()),
//         rt_policy_profile_(std::make_unique<PolicyProfile>(Frozen())),
//         iteration_(1),
//         rt_weight_(rt_weight),
//         T_(T) {}

//   Policy* Strategy(size_t player) const override final {
//     return learners_[player].get();
//   }

//   // print std::vector<double>
//   template <typename T>
//   void print_vector(const std::vector<T>& vec) const {
//     for (size_t i = 0; i < vec.size() && i < 10; ++i) {
//       std::cout << vec[i] << " ";
//     }
//     std::cout << std::endl;
//   }
//   // print template value first 10 entries of a map
//   template <typename T>
//   void print_map(const T& map) const {
//     size_t count = 0;
//     for (const auto& [key, value] : map) {
//       if (count++ < 10) {
//         std::cout << "Key: " << key << ", Value: ";
//         print_vector(value);
//       } else {
//         break;
//       }
//     }
//   }
//   Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
//                         const PolicyProfile& compatriots) override final {
//     MapPolicy policy=ToMapPolicy(root);
//     const auto& rt_value_map = ComputeRTValue(root, policy);
//     double avg = 0.0;
//     //print rt value map at second iteration
//     // if (iteration_ == 5) {
//     //   std::cout << "RT Value Map Sample (first 10 entries):" << std::endl;
//     //   print_map(rt_value_map); // debug
//     // }
//     for (int player = 0; player < root.NumPlayers(); ++player) {
//       auto [v, initial_info_states, cf_value_tree, _] =
//           evaluators_[player].ComputeCfValueTreeEvaluation(
//               root, compatriots.WithSubstitute(Strategy(player), player),
//               sampler);
//       // Compute the reward-transformed CFV.
//       const auto rt_cf_value_tree=ComputeRTcfValueTree(std::move(*cf_value_tree), rt_value_map);
//       learners_[player]->Update(initial_info_states, rt_cf_value_tree);
//       // Update the mean over all players; equivalent to avg = (avg*player + v)/(player+1).

//       avg += (v - avg) / (player + 1.0);
//     }
//     ++iteration_; // Increment the iteration counter.
//     // Update the reward-transformation policy every T iterations.
//     if (iteration_ % T_ == 0) {
//       rt_policy_profile_ = std::make_unique<PolicyProfile>(Frozen());
//     }
//     return avg;
//   }

//   Cfv UpdateAlternateAndReturnEv(DecisionPoint& root,
//                                  MccfrSampler& sampler) override final {
//                                   MapPolicy policy=ToMapPolicy(root);
//     const auto& rt_value_map = ComputeRTValue(root, policy);
//     double avg = 0.0;

//     for (int player = 0; player < root.NumPlayers(); ++player) {
//       auto [v, initial_info_states, cf_value_tree, _] =
//           evaluators_[player].ComputeCfValueTreeEvaluation(
//               root, PolicyRefProfile(learners_), sampler);
//       // Compute the reward-transformed CFV.
//       const auto rt_cf_value_tree=ComputeRTcfValueTree(std::move(*cf_value_tree), rt_value_map);
//       learners_[player]->Update(initial_info_states, rt_cf_value_tree);
//       avg += (v - avg) / (player + 1.0);
//     }
//     ++iteration_;
//     // Update the reward-transformation policy every T iterations.
//     if (iteration_ % T_ == 0) {
//       rt_policy_profile_ = std::make_unique<PolicyProfile>(Frozen());
//     }
//     return avg;
//   }

//   PolicyProfile Frozen() const override final { return Clone(learners_); }

//   // Recursively compute reward-transformation values for all states and return an InfoStateUvm<std::vector<double>> pointer.
//   InfoStateUvm<std::vector<double>> ComputeRTValue(DecisionPoint& root, Policy& policy) const {
//     InfoStateUvm<std::vector<double>> rt_value_map;
//     ForEachState(root, [&rt_value_map, &policy, this](const DecisionPoint& decision_point){
//       // Skip chance and terminal nodes.
//       if (decision_point.PlayerToAct() == -1) return;
//       //get policy and rt_policy at node
//       assert(decision_point.OpenSpielStatePtr());
//       const std::vector<double> policy_state=
//       policy.Response(*decision_point.OpenSpielStatePtr()); const std::vector<double>
//       rt_policy_state= rt_policy_profile_->Response(*decision_point.OpenSpielStatePtr());
//       // compute rt_value at node
//       std::vector<double> rt_value(policy_state.size(), 0.0);
//       for (size_t aidx=0; aidx<decision_point.NumActions(); ++aidx){
//         rt_value[aidx]= rt_weight_*(rt_policy_state[aidx]- policy_state[aidx]);
//       }
//       // insert or update rt value to map
//       const std::string& iss = decision_point.InformationStateStringRef();
//       rt_value_map[iss] = std::move(rt_value);
//     }, ALL_PLAYERS);
//     return rt_value_map;
//   }

//   // Compute the reward-transformed CFV tree and return an InfoStateUvm<CfValueTreeNode> pointer.
//   InfoStateUvm<CfValueTreeNode> ComputeRTcfValueTree(InfoStateUvm<CfValueTreeNode> cf_value_tree,
//                                                     const InfoStateUvm<std::vector<double>>&
//                                                     rt_value_map) const {
//     auto rt_cf_value_tree = std::move(cf_value_tree); // move instead of copy
//     for (auto& [iss, node] : rt_cf_value_tree) {
//       // find rt value at iss
//       assert(rt_value_map.contains(iss));
//       const auto& rt_value = rt_value_map.at(iss);
//       assert(rt_value.size() == node.cf_values_.Size());
//       // update rt cf values at node
//       for (size_t aidx = 0; aidx < node.cf_values_.Size(); ++aidx) {
//         node.cf_values_[aidx] += rt_value[aidx];
//       }
//     }
//     return rt_cf_value_tree;
//   }
//  private:
//  // Create one PolicyCfValueTreeEvaluator per player to record that player's CFV tree.
//   std::vector<PolicyCfValueTreeEvaluator> NewEvaluators() {
//     std::vector<PolicyCfValueTreeEvaluator> l;
//     for (size_t i = 0; i < learners_.size(); ++i) {
//       l.emplace_back(i); // = l.push_back(PolicyCfValueTreeEvaluator(i));
//     }
//     return l;
//   }
//  private:
//   std::vector<CfValueTreeLearnerPtr> learners_;
//   std::vector<PolicyCfValueTreeEvaluator> evaluators_;
//   double rt_weight_;
//   std::unique_ptr<PolicyProfile> rt_policy_profile_; // rt policy, update for each T iterations
//   size_t T_;
//   size_t iteration_;
// };

// Reward-transformation CFR profile inheriting AdaptiveProfile.
class PerturbedCfTreeLearnerProfile : public AdaptiveProfile {
 public:
  PerturbedCfTreeLearnerProfile(std::vector<CfValueTreeLearnerPtr>&& learners, double epsilon)
      : learners_(std::move(learners)),
        evaluators_(NewEvaluators()),
        // rt_policy_profile_(std::make_unique<PolicyProfile>(Frozen())),
        iteration_(1),
        epsilon_(epsilon) {}

  Policy* Strategy(size_t player) const override final { return learners_[player].get(); }

  // Policy* AverageStrategy(size_t player) const {
  //   return learners_[player].get();
  // }

  // print std::vector<double>
  template <typename T>
  void print_vector(const std::vector<T>& vec) const {
    for (size_t i = 0; i < vec.size() && i < 10; ++i) {
      std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
  }
  // print template value first 10 entries of a map
  template <typename T>
  void print_map(const T& map) const {
    size_t count = 0;
    for (const auto& [key, value] : map) {
      if (count++ < 10) {
        std::cout << "Key: " << key << ", Value: ";
        print_vector(value);
      } else {
        break;
      }
    }
  }

  Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
                        const PolicyProfile& compatriots) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      auto [v, initial_info_states, cf_value_tree, _] =
          evaluators_[player].ComputeCfValueTreeEvaluation(
              root, compatriots.WithSubstitute(Strategy(player), player), sampler);

      // Update the policy.
      learners_[player]->Update(initial_info_states, *cf_value_tree, epsilon_);
      // Update the mean over all players; equivalent to avg = (avg*player + v)/(player+1).
      avg += (v - avg) / (player + 1.0);
    }
    ++iteration_;  // Increment the iteration counter.
    return avg;
  }

  // Get the maximum phi regret.
  // std::tuple<double, double, double, double> ReachedPhiRegret(
  //     DecisionPoint& root, MccfrSampler& sampler) override final {
  //   double max_full_phi_regret_fus = 0.0;
  //   double max_full_phi_regret = 0.0;
  //   double max_phi_regret = 0.0;
  //   double sum_phi_regret = 0.0;

  //   for (int player = 0; player < root.NumPlayers(); ++player) {
  //     auto [v, initial_info_states, cf_value_tree, _] =
  //         evaluators_[player].ComputeCfValueTreeEvaluation(root, PolicyRefProfile(learners_),
  //                                                          sampler);  // Compute the CFV tree using the average policy.
  //     // Debug: print the CFV tree.
  //     // std::cout<< "player " << player << " cfv tree: " << std::endl;
  //     // for (const auto& [iss, node] : *cf_value_tree) {
  //     //   // Print the information-state string.
  //     //   std::cout << "InfoState: " << iss << std::endl;
  //     //   // Print the importance-weighted reach probability.
  //     //   std::cout << "IWRP: " << node.iwrp_ << std::endl;
  //     //   // Print the CFV.
  //     //   std::cout << "CfValues: ";
  //     //   for (size_t aidx = 0; aidx < node.cf_values_.Size(); ++aidx) {
  //     //     std::cout << node.cf_values_[aidx] << " ";
  //     //   }
  //     //   std::cout << std::endl;
  //     // }
  //     auto [player_max_full_phi_regret_fus, player_max_full_phi_regret, player_max_phi_regret,
  //           player_sum_phi_regret] =
  //         learners_[player]->LearnerReachedPhiRegret(initial_info_states, *cf_value_tree);
  //     if (player_max_phi_regret > max_phi_regret) {
  //       max_phi_regret = player_max_phi_regret;
  //     }
  //     sum_phi_regret += player_sum_phi_regret;
  //     if (player_max_full_phi_regret > max_full_phi_regret) {
  //       max_full_phi_regret = player_max_full_phi_regret;
  //     }
  //   }
  //   return {max_full_phi_regret_fus, max_full_phi_regret, max_phi_regret, sum_phi_regret};
  // }

  // Get phiRegrets for all players.
  std::tuple<double, double, double, double> PhiRegrets() const override final {
    double max_full_phi_regret_fus = 0.0;
    double max_full_phi_regret = 0.0;
    double max_phi_regret = 0.0;
    double sum_phi_regret = 0.0;

    for (int player = 0; player < learners_.size(); ++player) {
      auto [player_max_full_phi_regret_fus, player_max_full_phi_regret, player_max_phi_regret,
            player_sum_phi_regret] = learners_[player]->PhiRegrets();
      if (player_max_full_phi_regret_fus > max_full_phi_regret_fus) {
        max_full_phi_regret_fus = player_max_full_phi_regret_fus;
      }
      if (player_max_full_phi_regret > max_full_phi_regret) {
        max_full_phi_regret = player_max_full_phi_regret;
      }
      if (player_max_phi_regret > max_phi_regret) {
        max_phi_regret = player_max_phi_regret;
      }
      sum_phi_regret += player_sum_phi_regret;
    }
    return {max_full_phi_regret_fus, max_full_phi_regret, max_phi_regret, sum_phi_regret};
  }

  // Changing epsilon changes no cached object: the payoff transformation is
  // constructed from epsilon_ in every learner update.
  void ShrinkEpsilon(double new_epsilon) override final {
    if (!std::isfinite(new_epsilon) || new_epsilon < 0.0) {
      throw std::invalid_argument("epsilon must be finite and non-negative");
    }
    epsilon_ = new_epsilon;
  }

  double Epsilon() const override final { return epsilon_; }

  StageRegretMetrics StageMetrics(double w_numerical_eps) const override final {
    StageRegretMetrics metrics;
    for (size_t player = 0; player < learners_.size(); ++player) {
      auto player_metrics = learners_[player]->StageMetrics(w_numerical_eps);
      for (auto& exposure : player_metrics.exposure_entries) {
        exposure.player = static_cast<int>(player);
      }
      if (player_metrics.has_bottleneck) {
        player_metrics.bottleneck_player = static_cast<int>(player);
      }
      metrics.Merge(player_metrics);
    }
    metrics.Finalize();
    return metrics;
  }

  void ResetStageEvaluation() override final {
    for (auto& learner : learners_) learner->ResetStageEvaluation();
  }

  void ResetSolverRegrets() override final {
    for (auto& learner : learners_) learner->ResetSolverRegrets();
  }

  void SetCfrPlus(bool enabled) override final {
    for (auto& learner : learners_) learner->SetCfrPlus(enabled);
  }

  double SolverRegretL1() const override final {
    double total = 0.0;
    for (const auto& learner : learners_) total += learner->SolverRegretL1();
    return total;
  }

  double StageOriginalRegretL1() const override final {
    double total = 0.0;
    for (const auto& learner : learners_) total += learner->StageOriginalRegretL1();
    return total;
  }

  double StageExposureSum() const override final {
    double total = 0.0;
    for (const auto& learner : learners_) total += learner->StageExposureSum();
    return total;
  }

  Cfv UpdateAlternateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler) override final {
    // MapPolicy policy=ToMapPolicy(root);
    // const auto& rt_value_map = ComputePerturbedValue(root, policy);
    double avg = 0.0;

    for (int player = 0; player < root.NumPlayers(); ++player) {
      auto [v, initial_info_states, cf_value_tree, _] =
          evaluators_[player].ComputeCfValueTreeEvaluation(root, PolicyRefProfile(learners_),
                                                           sampler);
      // Compute the perturbed CFV.
      // const auto rt_cf_value_tree=ComputePerturbedcfValueTree(std::move(*cf_value_tree));
      learners_[player]->Update(initial_info_states, *cf_value_tree, epsilon_);
      avg += (v - avg) / (player + 1.0);
    }
    ++iteration_;

    return avg;
  }

  PolicyProfile Frozen() const override final { return Clone(learners_); }

  // Compute the perturbed CFV tree and return an InfoStateUvm<CfValueTreeNode> pointer.
  // InfoStateUvm<CfValueTreeNode> ComputePerturbedcfValueTree(InfoStateUvm<CfValueTreeNode>
  // cf_value_tree) const {
  //   auto p_cf_value_tree = std::move(cf_value_tree); // move instead of copy
  //   for (auto& [iss, node] : p_cf_value_tree) {
  //     // expected perturbed value
  //     // A more general epsilon could be a vector of action-specific perturbations, but that is unnecessary here.
  //     double p_value = 0.0;
  //     for (size_t aidx = 0; aidx < node.cf_values_.Size(); ++aidx) {
  //       p_value += node.cf_values_[aidx]*epsilon_;
  //     }
  //     for (size_t aidx = 0; aidx < node.cf_values_.Size(); ++aidx) {
  //       node.cf_values_[aidx] = (1-node.cf_values_.Size()*epsilon_)*node.cf_values_[aidx] +
  //       p_value;
  //     }
  //   }
  //   return p_cf_value_tree;
  // }

 private:
  // Create one PolicyCfValueTreeEvaluator per player to record that player's CFV tree.
  std::vector<PolicyCfValueTreeEvaluator> NewEvaluators() {
    std::vector<PolicyCfValueTreeEvaluator> l;
    for (size_t i = 0; i < learners_.size(); ++i) {
      l.emplace_back(i);  // = l.push_back(PolicyCfValueTreeEvaluator(i));
    }
    return l;
  }

 private:
  std::vector<CfValueTreeLearnerPtr> learners_;
  std::vector<PolicyCfValueTreeEvaluator> evaluators_;
  double epsilon_;
  // std::unique_ptr<PolicyProfile> rt_policy_profile_; // rt policy, update for each T iterations
  size_t iteration_;
};

// QBCE algorithm profile inheriting AdaptiveProfile.
class QBCECfTreeLearnerProfile : public AdaptiveProfile {
 public:
  // learners_: vector of owning CfValueTreeLearnerPtr pointers, one per player.
  // evaluators_: one PolicyCfValueTreeEvaluator per player, computing that player's CFV tree
  // from the root, player policies, and sampler.
  QBCECfTreeLearnerProfile(std::vector<CfValueTreeLearnerPtr>&& learners)
      : learners_(std::move(learners)), evaluators_(NewEvaluators()) {}

  Policy* Strategy(size_t player) const override final { return learners_[player].get(); }
  // Update the policy and return the expected value.
  // compatriots: policy profile with the policy at player replaced by the current player's policy.
  Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
                        const PolicyProfile& compatriots) override final {
    double avg = 0.0;
    // Compute the current player's CFV tree at each player position for the subsequent policy update.
    for (int player = 0; player < root.NumPlayers(); ++player) {
      // Get the initial sibling information-state strings and CFV tree to update.
      const auto [v, initial_info_states, cf_value_tree, _] =
          evaluators_[player].ComputeCfValueTreeEvaluation(
              root, compatriots.WithSubstitute(Strategy(player), player), sampler);
      // Update the player's policy from the initial information-state strings and CFV tree using stack push/pop operations.
      learners_[player]->Update(initial_info_states, *cf_value_tree);
      // Update the average over all players with the running-mean formula.
      // avg=avg+(v-avg)/(player+1) is equivalent to avg=(avg*player+v)/(player+1).
      // The former is numerically more stable and avoids overflow from a large avg*player product.
      avg += (v - avg) / (player + 1.0);
    }
    return avg;
  }
  // Update players in turn and return the expected value.
  Cfv UpdateAlternateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto [v, initial_info_states, cf_value_tree, _] =
          evaluators_[player].ComputeCfValueTreeEvaluation(root, PolicyRefProfile(learners_),
                                                           sampler);
      learners_[player]->Update(initial_info_states, *cf_value_tree);
      avg += (v - avg) / (player + 1.0);
    }
    return avg;
  }
  // Equivalent to return PolicyProfile(Clone(learners_)), implicitly invoking the return type's constructor.
  PolicyProfile Frozen() const override final { return Clone(learners_); }

 private:
  // Create one PolicyCfValueTreeEvaluator per player to record that player's CFV tree.
  std::vector<PolicyCfValueTreeEvaluator> NewEvaluators() {
    std::vector<PolicyCfValueTreeEvaluator> l;
    for (size_t i = 0; i < learners_.size(); ++i) {
      l.emplace_back(i);  // = l.push_back(PolicyCfValueTreeEvaluator(i));
    }
    return l;
  }

 private:
  std::vector<CfValueTreeLearnerPtr> learners_;
  std::vector<PolicyCfValueTreeEvaluator> evaluators_;
  // size_t k_queries_;
  // bool sequential_;
  // bool causal_;
};

// Inherits AdaptiveProfile.
class MapPolicyAdaptiveProfile : public AdaptiveProfile {
 public:
  MapPolicyAdaptiveProfile(size_t num_players) : AdaptiveProfile(), profile_() {
    profile_.reserve(num_players);
    for (size_t i = 0; i < num_players; ++i) {
      profile_.emplace_back(new MapPolicy());
    }
  }

  Policy* Strategy(size_t player) const override final { return profile_[player].get(); }
  PolicyProfile Frozen() const override final { return Clone(profile_); }

 protected:
  std::vector<MapPolicyPtr> profile_;
};

// Inherits MapPolicyAdaptiveProfile.
class PolicyIterationProfile : public MapPolicyAdaptiveProfile {
 public:
  PolicyIterationProfile(size_t num_players) : MapPolicyAdaptiveProfile(num_players) {}

  Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
                        const PolicyProfile& compatriots) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto [v, _] =
          PolicyValue(root, player, compatriots.WithSubstitute(Strategy(player), player), sampler);
      profile_[player].reset(
          new MapPolicy(BestResponse(compatriots).Policy(root, -player - 1).first));
      avg += (v - avg) / (player + 1.0);
    }
    return avg;
  }

  Cfv UpdateAlternateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto [v, _] = PolicyValue(root, player, PolicyRefProfile(profile_), sampler);
      profile_[player].reset(
          new MapPolicy(BestResponse(PolicyRefProfile(profile_)).Policy(root, -player - 1).first));
      avg += (v - avg) / (player + 1.0);
    }
    return avg;
  }
};

class AntiPolicyIterationProfile : public MapPolicyAdaptiveProfile {
 public:
  AntiPolicyIterationProfile(size_t num_players)
      : MapPolicyAdaptiveProfile(num_players), policy_iteration_(num_players) {}

  Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
                        const PolicyProfile& compatriots) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto [v, _] =
          PolicyValue(root, player, compatriots.WithSubstitute(Strategy(player), player), sampler);
      avg += (v - avg) / (player + 1.0);
    }
    policy_iteration_.UpdateAndReturnEv(root, sampler, compatriots);
    const auto pi_profile = policy_iteration_.Frozen();

    for (int player = 0; player < root.NumPlayers(); ++player) {
      profile_[player].reset(
          new MapPolicy(BestResponse(pi_profile).Policy(root, -player - 1).first));
    }
    return avg;
  }

  Cfv UpdateAlternateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto [v, _] = PolicyValue(root, player, PolicyRefProfile(profile_), sampler);
      avg += (v - avg) / (player + 1.0);

      policy_iteration_.UpdateAndReturnEv(root, sampler, Frozen());
      profile_[player].reset(
          new MapPolicy(BestResponse(policy_iteration_.Frozen()).Policy(root, -player - 1).first));
    }
    return avg;
  }

 private:
  PolicyIterationProfile policy_iteration_;
};

class BestResponseProfile : public MapPolicyAdaptiveProfile {
 public:
  BestResponseProfile(size_t num_players) : MapPolicyAdaptiveProfile(num_players) {}

  Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
                        const PolicyProfile& compatriots) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      auto [policy, v] = BestResponse(compatriots).Policy(root, -player - 1);
      profile_[player].reset(new MapPolicy(std::move(policy)));
      avg += (v - avg) / (player + 1.0);
    }
    return avg;
  }

  Cfv UpdateAlternateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      auto [policy, v] = BestResponse(PolicyRefProfile(profile_)).Policy(root, -player - 1);
      profile_[player].reset(new MapPolicy(std::move(policy)));
      avg += (v - avg) / (player + 1.0);
    }
    return avg;
  }
};

class FictitiousPlayProfile : public MapPolicyAdaptiveProfile {
 public:
  FictitiousPlayProfile(size_t num_players)
      : MapPolicyAdaptiveProfile(num_players), compatriot_empirical_play_() {}

  Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
                        const PolicyProfile& compatriots) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto [v, _] =
          PolicyValue(root, player, compatriots.WithSubstitute(Strategy(player), player), sampler);
      avg += (v - avg) / (player + 1.0);
    }
    compatriot_empirical_play_.Avg(compatriots, root);

    for (int player = 0; player < root.NumPlayers(); ++player) {
      profile_[player].reset(
          new MapPolicy(BestResponse(compatriot_empirical_play_).Policy(root, -player - 1).first));
    }
    return avg;
  }

  Cfv UpdateAlternateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto profile_ref = PolicyRefProfile(profile_);
      const auto [v, _] = PolicyValue(root, player, profile_ref, sampler);
      avg += (v - avg) / (player + 1.0);

      const int not_player = -player - 1;
      compatriot_empirical_play_.Avg(profile_ref, root, not_player);
      profile_[player].reset(
          new MapPolicy(BestResponse(compatriot_empirical_play_).Policy(root, not_player).first));
    }
    return avg;
  }

 private:
  MapPolicy compatriot_empirical_play_;
};
}  // namespace hr_edl

#endif  // HR_EDL_ADAPTIVE_PROFILE_H_
