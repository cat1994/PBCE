#include "hr_edl/policy_evaluation.h"

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "hr_edl/test_extra.h"

namespace hr_edl {
namespace test {
namespace {

void AlwaysFoldInLeduc() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("leduc_poker");
  CachedDecisionPoint decision_point(game->NewInitialState());
  NullSampler full_walk;

  const AlwaysZeroPolicy policy;
  {  // Player 1
    const auto [v1, regrets_and_reach_probs, num_decision_histories1] =
        PolicyRegretsAndReachProbs(decision_point, 0, policy, full_walk, 1);
    const auto [v2, regrets, num_decision_histories2] =
        PolicyCounterfactualRegrets(decision_point, 0, policy, full_walk);
    PolicyCfValueTreeEvaluator evaluator(0);
    const auto [v3, initial_info_states, cf_value_tree_ptr,
                num_decision_histories3] =
        evaluator.ComputeCfValueTreeEvaluation(decision_point, policy,
                                               full_walk);

    SPIEL_CHECK_FLOAT_EQ(v1, -2.08167e-17);
    SPIEL_CHECK_FLOAT_EQ(v1, v2);
    SPIEL_CHECK_FLOAT_EQ(v1, v3);
    SPIEL_CHECK_EQ(num_decision_histories1, 450);
    SPIEL_CHECK_EQ(num_decision_histories1, num_decision_histories2);
    // The PBCE value tree retains zero-probability opponent branches. Legacy
    // regret evaluators prune those branches, so only the full tree always
    // visits all 3,780 nonterminal Leduc histories (as under full support).
    SPIEL_CHECK_EQ(num_decision_histories3, 3780);
    SPIEL_CHECK_EQ(regrets_and_reach_probs.first.size(), regrets.size());
    // Check coverage against an independent traversal of all information states.
    SPIEL_CHECK_EQ(cf_value_tree_ptr->size(), NumStates(decision_point, 0));
    for (const auto& [iss, regrets2] : regrets_and_reach_probs.first) {
      SPIEL_CHECK_FLOAT_EQ(regrets2.ev_, regrets.at(iss).ev_);
      SPIEL_CHECK_FLOAT_EQ(regrets2.ev_,
                           cf_value_tree_ptr->at(iss).cf_values_.ev_);
      SPIEL_CHECK_EQ(regrets2.Size(), regrets.at(iss).Size());
      SPIEL_CHECK_EQ(regrets2.Size(),
                     cf_value_tree_ptr->at(iss).cf_values_.Size());
      for (size_t i = 0; i < regrets2.Size(); ++i) {
        SPIEL_CHECK_FLOAT_EQ(regrets2[i], regrets.at(iss)[i]);
        SPIEL_CHECK_FLOAT_EQ(regrets2[i],
                             cf_value_tree_ptr->at(iss).cf_values_[i]);
      }
    }
    // Check that cf_value_tree is actually a complete tree.
    SPIEL_CHECK_GT(initial_info_states.size(), 0);
    {
      size_t num_info_states = 0;
      std::vector<std::string> state_stack = initial_info_states;
      state_stack.reserve(cf_value_tree_ptr->size());
      while (state_stack.size() > 0) {
        const std::string iss = state_stack.back();
        state_stack.pop_back();
        ++num_info_states;

        const auto& _ = cf_value_tree_ptr->at(iss);
        const auto& cf_values = _.cf_values_;
        const auto& child_keys = _.child_keys_;
        const auto legacy_node = regrets.find(iss);
        if (legacy_node != regrets.end()) {
          const auto& x_cf_values = legacy_node->second;
          SPIEL_CHECK_FLOAT_EQ(x_cf_values.ev_, cf_values.ev_);
          SPIEL_CHECK_EQ(x_cf_values.v_.size(), cf_values.v_.size());
          for (size_t i = 0; i < x_cf_values.v_.size(); ++i) {
            SPIEL_CHECK_FLOAT_EQ(x_cf_values.v_[i], cf_values.v_[i]);
          }
        } else {
          // Nodes pruned by the legacy evaluator have zero counterfactual reach.
          // The PBCE tree must retain their structure with zero weighted values.
          SPIEL_CHECK_FLOAT_EQ(_.iwrp_, 0.0);
          SPIEL_CHECK_FLOAT_EQ(cf_values.ev_, 0.0);
          for (double value : cf_values.v_) SPIEL_CHECK_FLOAT_EQ(value, 0.0);
        }
        for (const auto& action_children : child_keys) {
          for (const auto& k : action_children) state_stack.push_back(k);
        }
      }
      SPIEL_CHECK_EQ(num_info_states, cf_value_tree_ptr->size());
    }
    {
      const auto [v3, num_decision_histories3] =
          PolicyValue(decision_point, 0, policy, full_walk);
      SPIEL_CHECK_FLOAT_EQ(v3, v1);
      SPIEL_CHECK_EQ(num_decision_histories3, 300);

      const auto [v4, _, num_decision_histories4] =
          PolicyReachWeightedRegrets(decision_point, 0, policy, full_walk);
      SPIEL_CHECK_FLOAT_EQ(v4, v1);
      SPIEL_CHECK_EQ(num_decision_histories4, num_decision_histories1);
    }
  }
  {  // Player 2
    const auto [v1, regrets_and_reach_probs, num_decision_histories1] =
        PolicyRegretsAndReachProbs(decision_point, 1, policy, full_walk, 0);
    const auto [v2, regrets, num_decision_histories2] =
        PolicyCounterfactualRegrets(decision_point, 1, policy, full_walk);
    PolicyCfValueTreeEvaluator evaluator(1);
    const auto [v3, initial_info_states, cf_value_tree_ptr,
                num_decision_histories3] =
        evaluator.ComputeCfValueTreeEvaluation(decision_point, policy,
                                               full_walk);

    SPIEL_CHECK_FLOAT_EQ(v1, 2.08167e-17);
    SPIEL_CHECK_FLOAT_EQ(v1, v2);
    SPIEL_CHECK_FLOAT_EQ(v1, v3);
    SPIEL_CHECK_EQ(num_decision_histories1, 450);
    SPIEL_CHECK_EQ(num_decision_histories1, num_decision_histories2);
    // The PBCE value tree retains zero-probability opponent branches. Legacy
    // regret evaluators prune those branches, so only the full tree always
    // visits all 3,780 nonterminal Leduc histories (as under full support).
    SPIEL_CHECK_EQ(num_decision_histories3, 3780);
    SPIEL_CHECK_EQ(regrets_and_reach_probs.first.size(), regrets.size());
    // Check coverage against an independent traversal of all information states.
    SPIEL_CHECK_EQ(cf_value_tree_ptr->size(), NumStates(decision_point, 1));
    for (const auto& [iss, regrets2] : regrets_and_reach_probs.first) {
      SPIEL_CHECK_FLOAT_EQ(regrets2.ev_, regrets.at(iss).ev_);
      SPIEL_CHECK_FLOAT_EQ(regrets2.ev_,
                           cf_value_tree_ptr->at(iss).cf_values_.ev_);
      SPIEL_CHECK_EQ(regrets2.Size(), regrets.at(iss).Size());
      SPIEL_CHECK_EQ(regrets2.Size(),
                     cf_value_tree_ptr->at(iss).cf_values_.Size());
      for (size_t i = 0; i < regrets2.Size(); ++i) {
        SPIEL_CHECK_FLOAT_EQ(regrets2[i], regrets.at(iss)[i]);
        SPIEL_CHECK_FLOAT_EQ(regrets2[i],
                             cf_value_tree_ptr->at(iss).cf_values_[i]);
      }
    }
    // Check that cf_value_tree is actually a complete tree.
    SPIEL_CHECK_GT(initial_info_states.size(), 0);
    {
      size_t num_info_states = 0;
      std::vector<std::string> state_stack = initial_info_states;
      state_stack.reserve(cf_value_tree_ptr->size());
      while (state_stack.size() > 0) {
        const std::string iss = state_stack.back();
        state_stack.pop_back();
        ++num_info_states;

        const auto& _ = cf_value_tree_ptr->at(iss);
        const auto& cf_values = _.cf_values_;
        const auto& child_keys = _.child_keys_;
        const auto legacy_node = regrets.find(iss);
        if (legacy_node != regrets.end()) {
          const auto& x_cf_values = legacy_node->second;
          SPIEL_CHECK_FLOAT_EQ(x_cf_values.ev_, cf_values.ev_);
          SPIEL_CHECK_EQ(x_cf_values.v_.size(), cf_values.v_.size());
          for (size_t i = 0; i < x_cf_values.v_.size(); ++i) {
            SPIEL_CHECK_FLOAT_EQ(x_cf_values.v_[i], cf_values.v_[i]);
          }
        } else {
          // Nodes pruned by the legacy evaluator have zero counterfactual reach.
          // The PBCE tree must retain their structure with zero weighted values.
          SPIEL_CHECK_FLOAT_EQ(_.iwrp_, 0.0);
          SPIEL_CHECK_FLOAT_EQ(cf_values.ev_, 0.0);
          for (double value : cf_values.v_) SPIEL_CHECK_FLOAT_EQ(value, 0.0);
        }
        for (const auto& action_children : child_keys) {
          for (const auto& k : action_children) state_stack.push_back(k);
        }
      }
      SPIEL_CHECK_EQ(num_info_states, cf_value_tree_ptr->size());
    }
    {
      const auto [v3, num_decision_histories3] =
          PolicyValue(decision_point, 1, policy, full_walk);
      SPIEL_CHECK_FLOAT_EQ(v3, v1);
      SPIEL_CHECK_EQ(num_decision_histories3, 300);

      const auto [v4, _, num_decision_histories4] =
          PolicyReachWeightedRegrets(decision_point, 1, policy, full_walk);
      SPIEL_CHECK_FLOAT_EQ(v4, v1);
      SPIEL_CHECK_EQ(num_decision_histories4, num_decision_histories1);
    }
  }
}

void AlwaysRaiseInLeduc() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("leduc_poker");
  CachedDecisionPoint decision_point(game->NewInitialState());
  NullSampler full_walk;

  const AlwaysMaxActionPolicy policy;
  {  // Player 1
    const auto [v1, regrets_and_reach_probs, num_decision_histories1] =
        PolicyRegretsAndReachProbs(decision_point, 0, policy, full_walk, 1);
    const auto [v2, regrets, num_decision_histories2] =
        PolicyCounterfactualRegrets(decision_point, 0, policy, full_walk);
    PolicyCfValueTreeEvaluator evaluator(0);
    const auto [v3, initial_info_states, cf_value_tree_ptr,
                num_decision_histories3] =
        evaluator.ComputeCfValueTreeEvaluation(decision_point, policy,
                                               full_walk);

    SPIEL_CHECK_FLOAT_EQ(v1, -1.11022e-16);
    SPIEL_CHECK_FLOAT_EQ(v1, v2);
    SPIEL_CHECK_FLOAT_EQ(v1, v3);
    SPIEL_CHECK_EQ(num_decision_histories1, 2340);
    SPIEL_CHECK_EQ(num_decision_histories1, num_decision_histories2);
    // The PBCE value tree retains zero-probability opponent branches. Legacy
    // regret evaluators prune those branches, so only the full tree always
    // visits all 3,780 nonterminal Leduc histories (as under full support).
    SPIEL_CHECK_EQ(num_decision_histories3, 3780);
    SPIEL_CHECK_EQ(regrets_and_reach_probs.first.size(), regrets.size());
    // Check coverage against an independent traversal of all information states.
    SPIEL_CHECK_EQ(cf_value_tree_ptr->size(), NumStates(decision_point, 0));
    for (const auto& [iss, regrets2] : regrets_and_reach_probs.first) {
      SPIEL_CHECK_FLOAT_EQ(regrets2.ev_, regrets.at(iss).ev_);
      SPIEL_CHECK_FLOAT_EQ(regrets2.ev_,
                           cf_value_tree_ptr->at(iss).cf_values_.ev_);
      SPIEL_CHECK_EQ(regrets2.Size(), regrets.at(iss).Size());
      SPIEL_CHECK_EQ(regrets2.Size(),
                     cf_value_tree_ptr->at(iss).cf_values_.Size());
      for (size_t i = 0; i < regrets2.Size(); ++i) {
        SPIEL_CHECK_FLOAT_EQ(regrets2[i], regrets.at(iss)[i]);
        SPIEL_CHECK_FLOAT_EQ(regrets2[i],
                             cf_value_tree_ptr->at(iss).cf_values_[i]);
      }
    }
    // Check that cf_value_tree is actually a complete tree.
    SPIEL_CHECK_GT(initial_info_states.size(), 0);
    {
      size_t num_info_states = 0;
      std::vector<std::string> state_stack = initial_info_states;
      state_stack.reserve(cf_value_tree_ptr->size());
      while (state_stack.size() > 0) {
        const std::string iss = state_stack.back();
        state_stack.pop_back();
        ++num_info_states;

        const auto& _ = cf_value_tree_ptr->at(iss);
        const auto& cf_values = _.cf_values_;
        const auto& child_keys = _.child_keys_;
        const auto legacy_node = regrets.find(iss);
        if (legacy_node != regrets.end()) {
          const auto& x_cf_values = legacy_node->second;
          SPIEL_CHECK_FLOAT_EQ(x_cf_values.ev_, cf_values.ev_);
          SPIEL_CHECK_EQ(x_cf_values.v_.size(), cf_values.v_.size());
          for (size_t i = 0; i < x_cf_values.v_.size(); ++i) {
            SPIEL_CHECK_FLOAT_EQ(x_cf_values.v_[i], cf_values.v_[i]);
          }
        } else {
          // Nodes pruned by the legacy evaluator have zero counterfactual reach.
          // The PBCE tree must retain their structure with zero weighted values.
          SPIEL_CHECK_FLOAT_EQ(_.iwrp_, 0.0);
          SPIEL_CHECK_FLOAT_EQ(cf_values.ev_, 0.0);
          for (double value : cf_values.v_) SPIEL_CHECK_FLOAT_EQ(value, 0.0);
        }
        for (const auto& action_children : child_keys) {
          for (const auto& k : action_children) state_stack.push_back(k);
        }
      }
      SPIEL_CHECK_EQ(num_info_states, cf_value_tree_ptr->size());
    }
    {
      const auto [v3, num_decision_histories3] =
          PolicyValue(decision_point, 0, policy, full_walk);
      SPIEL_CHECK_FLOAT_EQ(v3, v1);
      SPIEL_CHECK_EQ(num_decision_histories3, 450);

      const auto [v4, _, num_decision_histories4] =
          PolicyReachWeightedRegrets(decision_point, 0, policy, full_walk);
      SPIEL_CHECK_FLOAT_EQ(v4, v1);
      SPIEL_CHECK_EQ(num_decision_histories4, num_decision_histories1);
    }
  }
  {  // Player 2
    const auto [v1, regrets_and_reach_probs, num_decision_histories1] =
        PolicyRegretsAndReachProbs(decision_point, 1, policy, full_walk, 0);
    const auto [v2, regrets, num_decision_histories2] =
        PolicyCounterfactualRegrets(decision_point, 1, policy, full_walk);
    PolicyCfValueTreeEvaluator evaluator(1);
    const auto [v3, initial_info_states, cf_value_tree_ptr,
                num_decision_histories3] =
        evaluator.ComputeCfValueTreeEvaluation(decision_point, policy,
                                               full_walk);

    SPIEL_CHECK_FLOAT_EQ(v1, 1.11022e-16);
    SPIEL_CHECK_FLOAT_EQ(v1, v2);
    SPIEL_CHECK_FLOAT_EQ(v1, v3);
    SPIEL_CHECK_EQ(num_decision_histories1, 810);
    SPIEL_CHECK_EQ(num_decision_histories1, num_decision_histories2);
    // The PBCE value tree retains zero-probability opponent branches. Legacy
    // regret evaluators prune those branches, so only the full tree always
    // visits all 3,780 nonterminal Leduc histories (as under full support).
    SPIEL_CHECK_EQ(num_decision_histories3, 3780);
    SPIEL_CHECK_EQ(regrets_and_reach_probs.first.size(), regrets.size());
    // Check coverage against an independent traversal of all information states.
    SPIEL_CHECK_EQ(cf_value_tree_ptr->size(), NumStates(decision_point, 1));
    for (const auto& [iss, regrets2] : regrets_and_reach_probs.first) {
      SPIEL_CHECK_FLOAT_EQ(regrets2.ev_, regrets.at(iss).ev_);
      SPIEL_CHECK_FLOAT_EQ(regrets2.ev_,
                           cf_value_tree_ptr->at(iss).cf_values_.ev_);
      SPIEL_CHECK_EQ(regrets2.Size(), regrets.at(iss).Size());
      SPIEL_CHECK_EQ(regrets2.Size(),
                     cf_value_tree_ptr->at(iss).cf_values_.Size());
      for (size_t i = 0; i < regrets2.Size(); ++i) {
        SPIEL_CHECK_FLOAT_EQ(regrets2[i], regrets.at(iss)[i]);
        SPIEL_CHECK_FLOAT_EQ(regrets2[i],
                             cf_value_tree_ptr->at(iss).cf_values_[i]);
      }
    }
    // Check that cf_value_tree is actually a complete tree.
    SPIEL_CHECK_GT(initial_info_states.size(), 0);
    {
      size_t num_info_states = 0;
      std::vector<std::string> state_stack = initial_info_states;
      state_stack.reserve(cf_value_tree_ptr->size());
      while (state_stack.size() > 0) {
        const std::string iss = state_stack.back();
        state_stack.pop_back();
        ++num_info_states;

        const auto& _ = cf_value_tree_ptr->at(iss);
        const auto& cf_values = _.cf_values_;
        const auto& child_keys = _.child_keys_;
        const auto legacy_node = regrets.find(iss);
        if (legacy_node != regrets.end()) {
          const auto& x_cf_values = legacy_node->second;
          SPIEL_CHECK_FLOAT_EQ(x_cf_values.ev_, cf_values.ev_);
          SPIEL_CHECK_EQ(x_cf_values.v_.size(), cf_values.v_.size());
          for (size_t i = 0; i < x_cf_values.v_.size(); ++i) {
            SPIEL_CHECK_FLOAT_EQ(x_cf_values.v_[i], cf_values.v_[i]);
          }
        } else {
          // Nodes pruned by the legacy evaluator have zero counterfactual reach.
          // The PBCE tree must retain their structure with zero weighted values.
          SPIEL_CHECK_FLOAT_EQ(_.iwrp_, 0.0);
          SPIEL_CHECK_FLOAT_EQ(cf_values.ev_, 0.0);
          for (double value : cf_values.v_) SPIEL_CHECK_FLOAT_EQ(value, 0.0);
        }
        for (const auto& action_children : child_keys) {
          for (const auto& k : action_children) state_stack.push_back(k);
        }
      }
      SPIEL_CHECK_EQ(num_info_states, cf_value_tree_ptr->size());
    }
    {
      const auto [v3, num_decision_histories3] =
          PolicyValue(decision_point, 1, policy, full_walk);
      SPIEL_CHECK_FLOAT_EQ(v3, v1);
      SPIEL_CHECK_EQ(num_decision_histories3, 450);

      const auto [v4, _, num_decision_histories4] =
          PolicyReachWeightedRegrets(decision_point, 1, policy, full_walk);
      SPIEL_CHECK_FLOAT_EQ(v4, v1);
      SPIEL_CHECK_EQ(num_decision_histories4, num_decision_histories1);
    }
  }
}

void UniformRandomInLeduc() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("leduc_poker");
  CachedDecisionPoint decision_point(game->NewInitialState());
  NullSampler full_walk;

  const MapPolicy policy = UniformRandomPolicy();
  {  // Player 1
    const auto [v1, regrets_and_reach_probs, num_decision_histories1] =
        PolicyRegretsAndReachProbs(decision_point, 0, policy, full_walk, 1);
    const auto [v2, regrets, num_decision_histories2] =
        PolicyCounterfactualRegrets(decision_point, 0, policy, full_walk);
    PolicyCfValueTreeEvaluator evaluator(0);
    const auto [v3, initial_info_states, cf_value_tree_ptr,
                num_decision_histories3] =
        evaluator.ComputeCfValueTreeEvaluation(decision_point, policy,
                                               full_walk);

    SPIEL_CHECK_FLOAT_EQ(v1, -0.078125);
    SPIEL_CHECK_FLOAT_EQ(v1, v2);
    SPIEL_CHECK_FLOAT_EQ(v1, v3);
    SPIEL_CHECK_EQ(num_decision_histories1, 3780);
    SPIEL_CHECK_EQ(num_decision_histories1, num_decision_histories2);
    // The PBCE value tree retains zero-probability opponent branches. Legacy
    // regret evaluators prune those branches, so only the full tree always
    // visits all 3,780 nonterminal Leduc histories (as under full support).
    SPIEL_CHECK_EQ(num_decision_histories3, 3780);
    SPIEL_CHECK_EQ(regrets_and_reach_probs.first.size(), regrets.size());
    // Check coverage against an independent traversal of all information states.
    SPIEL_CHECK_EQ(cf_value_tree_ptr->size(), NumStates(decision_point, 0));
    for (const auto& [iss, regrets2] : regrets_and_reach_probs.first) {
      SPIEL_CHECK_FLOAT_EQ(regrets2.ev_, regrets.at(iss).ev_);
      SPIEL_CHECK_FLOAT_EQ(regrets2.ev_,
                           cf_value_tree_ptr->at(iss).cf_values_.ev_);
      SPIEL_CHECK_EQ(regrets2.Size(), regrets.at(iss).Size());
      SPIEL_CHECK_EQ(regrets2.Size(),
                     cf_value_tree_ptr->at(iss).cf_values_.Size());
      for (size_t i = 0; i < regrets2.Size(); ++i) {
        SPIEL_CHECK_FLOAT_EQ(regrets2[i], regrets.at(iss)[i]);
        SPIEL_CHECK_FLOAT_EQ(regrets2[i],
                             cf_value_tree_ptr->at(iss).cf_values_[i]);
      }
    }
    // Check that cf_value_tree is actually a complete tree.
    SPIEL_CHECK_GT(initial_info_states.size(), 0);
    {
      size_t num_info_states = 0;
      std::vector<std::string> state_stack = initial_info_states;
      state_stack.reserve(cf_value_tree_ptr->size());
      while (state_stack.size() > 0) {
        const std::string iss = state_stack.back();
        state_stack.pop_back();
        ++num_info_states;

        const auto& _ = cf_value_tree_ptr->at(iss);
        const auto& cf_values = _.cf_values_;
        const auto& child_keys = _.child_keys_;
        const auto legacy_node = regrets.find(iss);
        if (legacy_node != regrets.end()) {
          const auto& x_cf_values = legacy_node->second;
          SPIEL_CHECK_FLOAT_EQ(x_cf_values.ev_, cf_values.ev_);
          SPIEL_CHECK_EQ(x_cf_values.v_.size(), cf_values.v_.size());
          for (size_t i = 0; i < x_cf_values.v_.size(); ++i) {
            SPIEL_CHECK_FLOAT_EQ(x_cf_values.v_[i], cf_values.v_[i]);
          }
        } else {
          // Nodes pruned by the legacy evaluator have zero counterfactual reach.
          // The PBCE tree must retain their structure with zero weighted values.
          SPIEL_CHECK_FLOAT_EQ(_.iwrp_, 0.0);
          SPIEL_CHECK_FLOAT_EQ(cf_values.ev_, 0.0);
          for (double value : cf_values.v_) SPIEL_CHECK_FLOAT_EQ(value, 0.0);
        }
        for (const auto& action_children : child_keys) {
          for (const auto& k : action_children) state_stack.push_back(k);
        }
      }
      SPIEL_CHECK_EQ(num_info_states, cf_value_tree_ptr->size());
    }
    {
      const auto [v3, num_decision_histories3] =
          PolicyValue(decision_point, 0, policy, full_walk);
      SPIEL_CHECK_FLOAT_EQ(v3, v1);
      SPIEL_CHECK_EQ(num_decision_histories3, 3780);

      const auto [v4, _, num_decision_histories4] =
          PolicyReachWeightedRegrets(decision_point, 0, policy, full_walk);
      SPIEL_CHECK_FLOAT_EQ(v4, v1);
      SPIEL_CHECK_EQ(num_decision_histories4, num_decision_histories1);
    }
  }
  {  // Player 2
    const auto [v1, regrets_and_reach_probs, num_decision_histories1] =
        PolicyRegretsAndReachProbs(decision_point, 1, policy, full_walk, 0);
    const auto [v2, regrets, num_decision_histories2] =
        PolicyCounterfactualRegrets(decision_point, 1, policy, full_walk);
    PolicyCfValueTreeEvaluator evaluator(1);
    const auto [v3, initial_info_states, cf_value_tree_ptr,
                num_decision_histories3] =
        evaluator.ComputeCfValueTreeEvaluation(decision_point, policy,
                                               full_walk);

    SPIEL_CHECK_FLOAT_EQ(v1, 0.078125);
    SPIEL_CHECK_FLOAT_EQ(v1, v2);
    SPIEL_CHECK_FLOAT_EQ(v1, v3);
    SPIEL_CHECK_EQ(num_decision_histories1, 3780);
    SPIEL_CHECK_EQ(num_decision_histories1, num_decision_histories2);
    // The PBCE value tree retains zero-probability opponent branches. Legacy
    // regret evaluators prune those branches, so only the full tree always
    // visits all 3,780 nonterminal Leduc histories (as under full support).
    SPIEL_CHECK_EQ(num_decision_histories3, 3780);
    SPIEL_CHECK_EQ(regrets_and_reach_probs.first.size(), regrets.size());
    // Check coverage against an independent traversal of all information states.
    SPIEL_CHECK_EQ(cf_value_tree_ptr->size(), NumStates(decision_point, 1));
    for (const auto& [iss, regrets2] : regrets_and_reach_probs.first) {
      SPIEL_CHECK_FLOAT_EQ(regrets2.ev_, regrets.at(iss).ev_);
      SPIEL_CHECK_FLOAT_EQ(regrets2.ev_,
                           cf_value_tree_ptr->at(iss).cf_values_.ev_);
      SPIEL_CHECK_EQ(regrets2.Size(), regrets.at(iss).Size());
      SPIEL_CHECK_EQ(regrets2.Size(),
                     cf_value_tree_ptr->at(iss).cf_values_.Size());
      for (size_t i = 0; i < regrets2.Size(); ++i) {
        SPIEL_CHECK_FLOAT_EQ(regrets2[i], regrets.at(iss)[i]);
        SPIEL_CHECK_FLOAT_EQ(regrets2[i],
                             cf_value_tree_ptr->at(iss).cf_values_[i]);
      }
    }
    // Check that cf_value_tree is actually a complete tree.
    SPIEL_CHECK_GT(initial_info_states.size(), 0);
    {
      size_t num_info_states = 0;
      std::vector<std::string> state_stack = initial_info_states;
      state_stack.reserve(cf_value_tree_ptr->size());
      while (state_stack.size() > 0) {
        const std::string iss = state_stack.back();
        state_stack.pop_back();
        ++num_info_states;

        const auto& _ = cf_value_tree_ptr->at(iss);
        const auto& cf_values = _.cf_values_;
        const auto& child_keys = _.child_keys_;
        const auto legacy_node = regrets.find(iss);
        if (legacy_node != regrets.end()) {
          const auto& x_cf_values = legacy_node->second;
          SPIEL_CHECK_FLOAT_EQ(x_cf_values.ev_, cf_values.ev_);
          SPIEL_CHECK_EQ(x_cf_values.v_.size(), cf_values.v_.size());
          for (size_t i = 0; i < x_cf_values.v_.size(); ++i) {
            SPIEL_CHECK_FLOAT_EQ(x_cf_values.v_[i], cf_values.v_[i]);
          }
        } else {
          // Nodes pruned by the legacy evaluator have zero counterfactual reach.
          // The PBCE tree must retain their structure with zero weighted values.
          SPIEL_CHECK_FLOAT_EQ(_.iwrp_, 0.0);
          SPIEL_CHECK_FLOAT_EQ(cf_values.ev_, 0.0);
          for (double value : cf_values.v_) SPIEL_CHECK_FLOAT_EQ(value, 0.0);
        }
        for (const auto& action_children : child_keys) {
          for (const auto& k : action_children) state_stack.push_back(k);
        }
      }
      SPIEL_CHECK_EQ(num_info_states, cf_value_tree_ptr->size());
    }
    {
      const auto [v3, num_decision_histories3] =
          PolicyValue(decision_point, 1, policy, full_walk);
      SPIEL_CHECK_FLOAT_EQ(v3, v1);
      SPIEL_CHECK_EQ(num_decision_histories3, 3780);

      const auto [v4, _, num_decision_histories4] =
          PolicyReachWeightedRegrets(decision_point, 1, policy, full_walk);
      SPIEL_CHECK_FLOAT_EQ(v4, v1);
      SPIEL_CHECK_EQ(num_decision_histories4, num_decision_histories1);
    }
  }
}
}  // namespace

}  // namespace test
}  // namespace hr_edl

using namespace hr_edl::test;

int main(int argc, char** argv) {
  RUN_TEST(AlwaysFoldInLeduc);
  RUN_TEST(AlwaysRaiseInLeduc);
  RUN_TEST(UniformRandomInLeduc);
}
