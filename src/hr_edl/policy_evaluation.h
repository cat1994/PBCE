#ifndef HR_EDL_POLICY_EVALUATION_H_
#define HR_EDL_POLICY_EVALUATION_H_

#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "hr_edl/decision_point.h"
#include "hr_edl/math.h"
#include "hr_edl/samplers.h"
#include "hr_edl/types.h"
#include "hr_edl/policy.h"

namespace hr_edl {

class PolicyRegretAndReachProbEvaluator {
 public:
  PolicyRegretAndReachProbEvaluator(int regret_player, const Policy& profile,
                                    MccfrSampler& sampler, size_t num_players,
                                    int reach_prob_player = -1)
      : regret_table_(),
        reach_probs_table_(),
        num_decision_histories_(0),
        regret_player_(regret_player),
        profile_(profile),
        sampler_(sampler),
        reach_prob_player_(reach_prob_player),
        reach_probabilities_(num_players, 1.0) {}
  virtual ~PolicyRegretAndReachProbEvaluator() = default;

  inline bool SaveRegrets(int current_player) const {
    return current_player == regret_player_;
  }
  inline bool SaveReachProbs(int current_player) const {
    return current_player == reach_prob_player_;
  }

  Cfv operator()(const open_spiel::State& state,
                 double chance_reach_importance_weight,
                 double player_sampling_prob);
  Cfv operator()(DecisionPoint& decision_point,
                 double chance_reach_importance_weight,
                 double player_sampling_prob);
  Cfv CounterfactualValue(DecisionPoint& decision_point,
                          double chance_reach_importance_weight,
                          double player_sampling_prob, int action_idx);

 public:
  InfoStateUvm<CfValues> regret_table_;
  InfoStateUvm<std::vector<double>> reach_probs_table_;
  int num_decision_histories_;

 private:
  double Return(DecisionPoint& decision_point) const {
    return decision_point.ReturnsRef()[regret_player_];
  }
  double CounterfactualReachProb() const {
    return ::hr_edl::CounterfactualReachProb(reach_probabilities_,
                                                         regret_player_);
  }
  Cfv CounterfactualValue(DecisionPoint& decision_point,
                          double chance_reach_importance_weight,
                          double player_sampling_prob, int action_idx,
                          double next_reach_prob);

 private:
  const int regret_player_;
  const Policy& profile_;
  MccfrSampler& sampler_;
  int reach_prob_player_;
  std::vector<double> reach_probabilities_;
};

std::tuple<Cfv,
           std::pair<InfoStateUvm<CfValues>, InfoStateUvm<std::vector<double>>>,
           int>
PolicyRegretsAndReachProbs(const open_spiel::State& root, int regret_player,
                           const Policy& profile, MccfrSampler& sampler,
                           int reach_prob_player = -1);

std::tuple<Cfv,
           std::pair<InfoStateUvm<CfValues>, InfoStateUvm<std::vector<double>>>,
           int>
PolicyRegretsAndReachProbs(DecisionPoint& root, int regret_player,
                           const Policy& profile, MccfrSampler& sampler,
                           int reach_prob_player = -1);

// Evaluate counterfactual policy regrets.
class PolicyRegretEvaluator {
 public:
  PolicyRegretEvaluator(int regret_player, const Policy& profile,
                        MccfrSampler& sampler, size_t num_players)
      : regret_table_(),
        num_decision_histories_(0),
        regret_player_(regret_player),
        profile_(profile),
        sampler_(sampler),
        reach_probabilities_(num_players, 1.0) {}
  virtual ~PolicyRegretEvaluator() = default;

  inline bool SaveRegrets(int current_player) const {
    return current_player == regret_player_;
  }

  Cfv ComputeCounterfactualRegrets(DecisionPoint& decision_point,
                                   double importance_weight);
  Cfv ComputeReachWeightedRegrets(DecisionPoint& decision_point,
                                  double importance_weight);
  Cfv CounterfactualValue(
      DecisionPoint& decision_point, double importance_weight, int action_idx,
      const std::function<Cfv(DecisionPoint&, double)>& backup);

 public:
  InfoStateUvm<CfValues> regret_table_;
  int num_decision_histories_;

 private:
  double Return(DecisionPoint& decision_point) const {
    return decision_point.ReturnsRef()[regret_player_];
  }
  double CounterfactualReachProb() const {
    return ::hr_edl::CounterfactualReachProb(reach_probabilities_,
                                                         regret_player_);
  }
  Cfv CounterfactualValue(
      DecisionPoint& decision_point, double importance_weight, int action_idx,
      double next_reach_prob,
      const std::function<Cfv(DecisionPoint&, double)>& backup);

 private:
  const int regret_player_;
  const Policy& profile_;
  MccfrSampler& sampler_;
  std::vector<double> reach_probabilities_;
};

std::tuple<Cfv, InfoStateUvm<CfValues>, int> PolicyCounterfactualRegrets(
    DecisionPoint& root, int regret_player, const Policy& profile,
    MccfrSampler& sampler);
std::tuple<Cfv, InfoStateUvm<CfValues>, int> PolicyReachWeightedRegrets(
    DecisionPoint& root, int regret_player, const Policy& profile,
    MccfrSampler& sampler);

class PolicyValueEvaluator {
 public:
  PolicyValueEvaluator(int player, const Policy& profile, MccfrSampler& sampler,
                       size_t num_players)
      : num_decision_histories_(0),
        player_(player),
        profile_(profile),
        sampler_(sampler) {}

  Cfv operator()(DecisionPoint& decision_point,
                 double importance_weighted_reach_prob, int action_idx); // Evaluate a specific action at the decision point.
  Cfv operator()(DecisionPoint& decision_point,
                 double importance_weighted_reach_prob); // Evaluate the expected return at the decision point.

 public:
  int num_decision_histories_;

 private:
  double Return(DecisionPoint& decision_point) const {
    return decision_point.ReturnsRef()[player_];
  }

 private:
  const int player_;
  const Policy& profile_;
  MccfrSampler& sampler_;
};

std::pair<Cfv, int> PolicyValue(DecisionPoint& root, int player,
                                const Policy& profile, MccfrSampler& sampler);

// Counterfactual-value tree result and its information-state node map.
struct CfValueTreeEvaluation {
  const Cfv ev_; // Expected return at the root.
  const std::vector<std::string> initial_info_states_; // Initial information-state keys of the target player.
  const InfoStateUvm<CfValueTreeNode>* cfv_nodes_; // Borrowed pointer to the counterfactual-value node map.
  const size_t num_histories_; // Number of decision histories traversed.
};

// Evaluate a policy and construct its counterfactual-value tree.
class PolicyCfValueTreeEvaluator {
 public:
  PolicyCfValueTreeEvaluator(int regret_player)
      : cf_value_tree_(),
        num_decision_histories_(0),
        regret_player_(regret_player) {} 
  virtual ~PolicyCfValueTreeEvaluator() = default;

  // Store regrets only for the designated regret player.
  bool SaveRegrets(int current_player) const {
    return current_player == regret_player_;
  }

  // Clear the previous evaluation.
  void Reset() {
    cf_value_tree_.clear();
    num_decision_histories_ = 0;
  }

  CfValueTreeEvaluation ComputeCfValueTreeEvaluation(DecisionPoint& root,
                                                     const Policy& profile,
                                                     MccfrSampler& sampler) { // Evaluate the supplied policy profile.
    Reset(); // Reset the value tree and history count for this evaluation.
    std::vector<std::string> initial_keys; // Initial information-state keys.
    const Cfv root_val =
        CounterfactualValue(initial_keys, root, profile, sampler, 1.0); // Start with importance-weighted reach probability one.
    return {root_val, std::move(initial_keys), &cf_value_tree_,
            num_decision_histories_};
  }

  // Evaluate artificial-root action zero to obtain the root counterfactual value.
  Cfv CounterfactualValue(std::vector<std::string>& siblings,
                          DecisionPoint& decision_point, const Policy& profile,
                          MccfrSampler& sampler,
                          double importance_weighted_reach_prob,
                          bool is_current_strategy=true) {
    return CounterfactualValue(siblings, decision_point, profile, sampler,
                               importance_weighted_reach_prob, 0);
  }

 public:
  InfoStateUvm<CfValueTreeNode> cf_value_tree_; // Map information-state keys to counterfactual-value nodes.
  size_t num_decision_histories_;

 private:
  double Return(DecisionPoint& decision_point) const {
    return decision_point.ReturnsRef()[regret_player_];
  }
  // Evaluate the counterfactual value of a specific action.
  Cfv CounterfactualValue(std::vector<std::string>& siblings,
                          DecisionPoint& decision_point, 
                          const Policy& profile,
                          MccfrSampler& sampler,
                          double importance_weighted_reach_prob,
                          int action_idx);
  Cfv ComputeCfValueTree(std::vector<std::string>& siblings,
                         DecisionPoint& decision_point, 
                         const Policy& profile,
                         MccfrSampler& sampler,
                         double importance_weighted_reach_prob);
 private:
  const int regret_player_; // Player whose regrets are evaluated.
};
}  // namespace hr_edl

#endif  // HR_EDL_POLICY_EVALUATION_H_
