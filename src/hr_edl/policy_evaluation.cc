#include "hr_edl/policy_evaluation.h"

#include <cassert>

using namespace open_spiel;

namespace hr_edl {

Cfv PolicyRegretAndReachProbEvaluator::operator()(
    const open_spiel::State& state, double chance_reach_iw,
    double player_sampling_prob) {
  if (state.IsTerminal()) {
    return state.Returns()[regret_player_] * chance_reach_iw *
           CounterfactualReachProb() / player_sampling_prob;
  } else if (state.IsChanceNode()) {
    Cfv state_value = 0.0;
    sampler_.SampleChanceOutcomes(
        state,
        [this, &state, &state_value, chance_reach_iw, player_sampling_prob](
            const ActionAndProb& outcome, double importance_weight) {
          state_value += (*this)(*state.Child(outcome.first),
                                 chance_reach_iw * importance_weight,
                                 player_sampling_prob);
        });
    return state_value;
  }

  ++num_decision_histories_;

  const int current_player = state.CurrentPlayer();
  const std::vector<open_spiel::Action> legal_actions =
      state.LegalActions(current_player);
  const std::vector<double>& policy = profile_.Response(state);
  const std::string info_state = state.InformationStateString(current_player);
  const int num_legal_actions = legal_actions.size();
  const double my_reach_prob = reach_probabilities_[current_player];

  if (SaveReachProbs(current_player)) {
    auto& reach_probs = GetOrCreate<std::string, std::vector<double>>(
        reach_probs_table_, info_state, [num_legal_actions]() {
          return std::vector<double>(num_legal_actions, 0.0);
        });
    for (int action_idx = 0; action_idx < num_legal_actions; ++action_idx) {
      const auto child_prob = policy[action_idx];

      if (child_prob > 0) {
        const auto reach_importance_weight =
            my_reach_prob / player_sampling_prob;
        reach_probs[action_idx] += reach_importance_weight * child_prob;
      }
    }
  }

  Cfv state_value = 0.0;
  if (SaveRegrets(current_player)) {
    double action_values[num_legal_actions];
    sampler_.SampleTargetPlayerActions(
        policy, [this, &action_values, &legal_actions, &state, chance_reach_iw,
                 player_sampling_prob, &state_value](
                    int action_idx, double action_prob, double sampling_prob) {
          const Cfv child_utility =
              (*this)(*state.Child(legal_actions[action_idx]), chance_reach_iw,
                      player_sampling_prob * sampling_prob);
          action_values[action_idx] = child_utility;
          const Cfv weighted_child_utility = action_prob * child_utility;
          state_value += weighted_child_utility;
        });
    auto& regrets = GetOrCreate<std::string, CfValues>(
        regret_table_, info_state, [num_legal_actions]() -> CfValues {
          return {CfActionValues(num_legal_actions, 0), 0};
        });
    regrets.ev_ += state_value;
    for (size_t a = 0; a < num_legal_actions; ++a) {
      regrets.v_[a] += action_values[a];
    }
  } else {
    sampler_.SampleExternalPlayerActions(
        policy, [this, my_reach_prob, chance_reach_iw, player_sampling_prob,
                 &legal_actions, current_player, &state_value, &state](
                    int action_idx, double action_prob, double sampling_prob) {
          const auto child_reach_probability = my_reach_prob * action_prob;

          if (!SaveReachProbs(current_player) || child_reach_probability > 0) {
            reach_probabilities_[current_player] = child_reach_probability;

            state_value +=
                (*this)(*state.Child(legal_actions[action_idx]),
                        chance_reach_iw, player_sampling_prob * sampling_prob);
          }
        });

    reach_probabilities_[current_player] = my_reach_prob;
  }
  return state_value;
}

Cfv PolicyRegretAndReachProbEvaluator::CounterfactualValue(
    DecisionPoint& decision_point, double chance_reach_iw,
    double player_sampling_prob, int action_idx) {
  Cfv v = 0;
  sampler_.SampleChanceOutcomes(
      decision_point.OutcomeProbabilities(action_idx),
      [this, action_idx, &decision_point, chance_reach_iw, player_sampling_prob,
       &v](int outcome, double outcome_prob, double outcome_sampling_prob) {
        decision_point.Apply(action_idx, outcome);
        v += (*this)(decision_point,
                     chance_reach_iw * outcome_prob / outcome_sampling_prob,
                     player_sampling_prob);
        decision_point.Undo();
      });
  return v;
}

Cfv PolicyRegretAndReachProbEvaluator::CounterfactualValue(
    DecisionPoint& decision_point, double chance_reach_iw,
    double player_sampling_prob, int action_idx, double next_reach_prob) {
  Cfv v = 0;
  sampler_.SampleChanceOutcomes(
      decision_point.OutcomeProbabilities(action_idx),
      [this, action_idx, &decision_point, chance_reach_iw, player_sampling_prob,
       next_reach_prob,
       &v](int outcome, double outcome_prob, double outcome_sampling_prob) {
        reach_probabilities_[decision_point.PlayerToAct()] = next_reach_prob;
        decision_point.Apply(action_idx, outcome);
        v += (*this)(decision_point,
                     chance_reach_iw * outcome_prob / outcome_sampling_prob,
                     player_sampling_prob);
        decision_point.Undo();
      });
  return v;
}

Cfv PolicyRegretAndReachProbEvaluator::operator()(DecisionPoint& decision_point,
                                                  double chance_reach_iw,
                                                  double player_sampling_prob) {
  if (decision_point.IsTerminal()) {
    return Return(decision_point) * chance_reach_iw *
           CounterfactualReachProb() / player_sampling_prob;
  }
  ++num_decision_histories_;

  const int current_player = decision_point.PlayerToAct();
  assert(decision_point.OpenSpielStatePtr());
  const std::vector<double> policy =
      profile_.Response(*decision_point.OpenSpielStatePtr());
  const std::string info_state = decision_point.InformationStateString();
  const int num_legal_actions = decision_point.NumActions();
  const double my_reach_prob = reach_probabilities_[current_player];

  if (SaveReachProbs(current_player)) {
    auto& reach_probs = GetOrCreate<std::string, std::vector<double>>(
        reach_probs_table_, info_state, [num_legal_actions]() {
          return std::vector<double>(num_legal_actions, 0.0);
        });
    for (int action_idx = 0; action_idx < num_legal_actions; ++action_idx) {
      reach_probs[action_idx] +=
          policy[action_idx] * my_reach_prob / player_sampling_prob;
    }
  }

  Cfv state_value = 0.0;
  if (SaveRegrets(current_player)) {
    double action_values[num_legal_actions];
    sampler_.SampleTargetPlayerActions(
        policy, [this, &decision_point, chance_reach_iw, player_sampling_prob,
                 &action_values, &state_value](
                    int action_idx, double action_prob, double sampling_prob) {
          const Cfv cfv = this->CounterfactualValue(
              decision_point, chance_reach_iw,
              player_sampling_prob * sampling_prob, action_idx);
          action_values[action_idx] = cfv;
          state_value += action_prob * cfv;
        });
    auto& regrets = GetOrCreate<std::string, CfValues>(
        regret_table_, info_state, [num_legal_actions]() -> CfValues {
          return {CfActionValues(num_legal_actions, 0), 0};
        });
    regrets.ev_ += state_value;
    for (size_t a = 0; a < num_legal_actions; ++a) {
      regrets.v_[a] += action_values[a];
    }
  } else {
    sampler_.SampleExternalPlayerActions(
        policy, [this, my_reach_prob, chance_reach_iw, player_sampling_prob,
                 current_player, &state_value, &decision_point](
                    int action_idx, double action_prob, double sampling_prob) {
          const double next_reach_probability = my_reach_prob * action_prob;
          if (!SaveReachProbs(current_player) || next_reach_probability > 0) {
            state_value +=
                CounterfactualValue(decision_point, chance_reach_iw,
                                    player_sampling_prob * sampling_prob,
                                    action_idx, next_reach_probability);
          }
        });
    reach_probabilities_[current_player] = my_reach_prob;
  }
  return state_value;
}

std::tuple<Cfv,
           std::pair<InfoStateUvm<CfValues>, InfoStateUvm<std::vector<double>>>,
           int>
PolicyRegretsAndReachProbs(const open_spiel::State& root, int regret_player,
                           const Policy& profile, MccfrSampler& sampler,
                           int reach_prob_player) {
  PolicyRegretAndReachProbEvaluator evaluator(
      regret_player, profile, sampler, root.NumPlayers(), reach_prob_player);
  const Cfv ev = evaluator(root, 1, 1);
  return {ev,
          {std::move(evaluator.regret_table_),
           std::move(evaluator.reach_probs_table_)},
          evaluator.num_decision_histories_};
}

std::tuple<Cfv,
           std::pair<InfoStateUvm<CfValues>, InfoStateUvm<std::vector<double>>>,
           int>
PolicyRegretsAndReachProbs(DecisionPoint& root, int regret_player,
                           const Policy& profile, MccfrSampler& sampler,
                           int reach_prob_player) {
  PolicyRegretAndReachProbEvaluator evaluator(
      regret_player, profile, sampler, root.NumPlayers(), reach_prob_player);
  const Cfv ev = evaluator.CounterfactualValue(root, 1.0, 1.0, 0);
  return {ev,
          {std::move(evaluator.regret_table_),
           std::move(evaluator.reach_probs_table_)},
          evaluator.num_decision_histories_};
}

Cfv PolicyRegretEvaluator::CounterfactualValue(
    DecisionPoint& decision_point, double importance_weight, int action_idx,
    const std::function<Cfv(DecisionPoint&, double)>& backup) {
  Cfv v = 0;
  sampler_.SampleChanceOutcomes(
      decision_point.OutcomeProbabilities(action_idx),
      [action_idx, &decision_point, importance_weight, &backup, &v](
          int outcome, double outcome_prob, double outcome_sampling_prob) {
        decision_point.Apply(action_idx, outcome);
        v += backup(decision_point,
                    importance_weight * outcome_prob / outcome_sampling_prob);
        decision_point.Undo();
      });
  return v;
}

Cfv PolicyRegretEvaluator::CounterfactualValue(
    DecisionPoint& decision_point, double importance_weight, int action_idx,
    double next_reach_prob,
    const std::function<Cfv(DecisionPoint&, double)>& backup) {
  Cfv v = 0;
  sampler_.SampleChanceOutcomes(
      decision_point.OutcomeProbabilities(action_idx),
      [this, action_idx, &decision_point, importance_weight, next_reach_prob,
       &backup,
       &v](int outcome, double outcome_prob, double outcome_sampling_prob) {
        reach_probabilities_[decision_point.PlayerToAct()] = next_reach_prob;
        decision_point.Apply(action_idx, outcome);
        v += backup(decision_point,
                    importance_weight * outcome_prob / outcome_sampling_prob);
        decision_point.Undo();
      });
  return v;
}

Cfv PolicyRegretEvaluator::ComputeCounterfactualRegrets(
    DecisionPoint& decision_point, double importance_weight) {
  if (decision_point.IsTerminal()) {
    return Return(decision_point) * importance_weight *
           CounterfactualReachProb();
  }
  ++num_decision_histories_;

  assert(decision_point.OpenSpielStatePtr());
  const std::vector<double> policy =
      profile_.Response(*decision_point.OpenSpielStatePtr());

  Cfv state_value = 0.0;
  if (SaveRegrets(decision_point.PlayerToAct())) {
    double action_values[decision_point.NumActions()];
    sampler_.SampleTargetPlayerActions(
        policy, [this, &decision_point, importance_weight, &action_values,
                 &state_value](int action_idx, double action_prob,
                               double sampling_prob) {
          const Cfv cfv = this->CounterfactualValue(
              decision_point, importance_weight / sampling_prob, action_idx,
              [this](DecisionPoint& next_dp, double next_iw) {
                return this->ComputeCounterfactualRegrets(next_dp, next_iw);
              });
          action_values[action_idx] = cfv;
          state_value += action_prob * cfv;
        });
    const std::string info_state = decision_point.InformationStateString();
    const int num_legal_actions = decision_point.NumActions();
    auto& regrets = GetOrCreate<std::string, CfValues>(
        regret_table_, info_state, [num_legal_actions]() -> CfValues {
          return {CfActionValues(num_legal_actions, 0), 0};
        });
    regrets.ev_ += state_value;
    for (size_t a = 0; a < num_legal_actions; ++a) {
      regrets.v_[a] += action_values[a];
    }
  } else {
    const double my_reach_prob =
        reach_probabilities_[decision_point.PlayerToAct()];
    sampler_.SampleExternalPlayerActions(
        policy,
        [this, my_reach_prob, importance_weight, &state_value, &decision_point](
            int action_idx, double action_prob, double sampling_prob) {
          const double next_reach_probability = my_reach_prob * action_prob;
          if (next_reach_probability > 0) {
            state_value += CounterfactualValue(
                decision_point, importance_weight / sampling_prob, action_idx,
                next_reach_probability,
                [this](DecisionPoint& next_dp, double next_iw) {
                  return this->ComputeCounterfactualRegrets(next_dp, next_iw);
                });
          }
        });
    reach_probabilities_[decision_point.PlayerToAct()] = my_reach_prob;
  }
  return state_value;
}

Cfv PolicyRegretEvaluator::ComputeReachWeightedRegrets(
    DecisionPoint& decision_point, double importance_weight) {
  if (decision_point.IsTerminal()) {
    return Return(decision_point) * importance_weight *
           CounterfactualReachProb();
  }
  ++num_decision_histories_;

  const int current_player = decision_point.PlayerToAct();
  assert(decision_point.OpenSpielStatePtr());
  const std::vector<double> policy =
      profile_.Response(*decision_point.OpenSpielStatePtr());
  const double my_reach_prob = reach_probabilities_[current_player];

  Cfv state_value = 0.0;
  if (SaveRegrets(current_player)) {
    double action_values[decision_point.NumActions()];
    sampler_.SampleTargetPlayerActions(
        policy, [this, &decision_point, importance_weight, my_reach_prob,
                 &action_values, &state_value](
                    int action_idx, double action_prob, double sampling_prob) {
          const Cfv cfv = this->CounterfactualValue(
              decision_point, importance_weight / sampling_prob, action_idx,
              my_reach_prob * action_prob,
              [this](DecisionPoint& next_dp, double next_iw) {
                return this->ComputeReachWeightedRegrets(next_dp, next_iw);
              });
          action_values[action_idx] = my_reach_prob * cfv;
          state_value += action_prob * cfv;
        });
    const std::string info_state = decision_point.InformationStateString();
    const size_t n = decision_point.NumActions();
    auto& regrets = GetOrCreate<std::string, CfValues>(
        regret_table_, info_state, [n]() -> CfValues {
          return {CfActionValues(n, 0), 0};
        });
    regrets.ev_ += my_reach_prob * state_value;
    for (size_t a = 0; a < n; ++a) {
      regrets.v_[a] += action_values[a];
    }
  } else {
    sampler_.SampleExternalPlayerActions(
        policy, [this, my_reach_prob, importance_weight, current_player,
                 &state_value, &decision_point](
                    int action_idx, double action_prob, double sampling_prob) {
          const double next_reach_probability = my_reach_prob * action_prob;
          if (next_reach_probability > 0) {
            state_value += CounterfactualValue(
                decision_point, importance_weight / sampling_prob, action_idx,
                next_reach_probability,
                [this](DecisionPoint& next_dp, double next_iw) {
                  return this->ComputeReachWeightedRegrets(next_dp, next_iw);
                });
          }
        });
  }
  reach_probabilities_[current_player] = my_reach_prob;
  return state_value;
}

std::tuple<Cfv, InfoStateUvm<CfValues>, int> PolicyCounterfactualRegrets(
    DecisionPoint& root, int regret_player, const Policy& profile,
    MccfrSampler& sampler) {
  PolicyRegretEvaluator evaluator(regret_player, profile, sampler,
                                  root.NumPlayers());
  const Cfv ev = evaluator.CounterfactualValue(
      root, 1.0, 0, [&evaluator](DecisionPoint& next_dp, double next_iw) {
        return evaluator.ComputeCounterfactualRegrets(next_dp, next_iw);
      });
  return {ev, std::move(evaluator.regret_table_),
          evaluator.num_decision_histories_};
}

std::tuple<Cfv, InfoStateUvm<CfValues>, int> PolicyReachWeightedRegrets(
    DecisionPoint& root, int regret_player, const Policy& profile,
    MccfrSampler& sampler) {
  PolicyRegretEvaluator evaluator(regret_player, profile, sampler,
                                  root.NumPlayers());
  const Cfv ev = evaluator.CounterfactualValue(
      root, 1.0, 0, [&evaluator](DecisionPoint& next_dp, double next_iw) {
        return evaluator.ComputeReachWeightedRegrets(next_dp, next_iw);
      });
  return {ev, std::move(evaluator.regret_table_),
          evaluator.num_decision_histories_};
}

Cfv PolicyValueEvaluator::operator()(DecisionPoint& decision_point,
                                     double importance_weighted_reach_prob,
                                     int action_idx) {
  Cfv v = 0;
  sampler_.SampleChanceOutcomes(
      decision_point.OutcomeProbabilities(action_idx),
      [this, action_idx, &decision_point, importance_weighted_reach_prob, &v](
          int outcome, double outcome_prob, double outcome_sampling_prob) {
        decision_point.Apply(action_idx, outcome);
        v += (*this)(decision_point, importance_weighted_reach_prob *
                                         outcome_prob / outcome_sampling_prob);
        decision_point.Undo();
      });
  return v;
}

Cfv PolicyValueEvaluator::operator()(DecisionPoint& decision_point,
                                     double importance_weighted_reach_prob) {
  if (decision_point.IsTerminal()) {
    return Return(decision_point) * importance_weighted_reach_prob;
  }
  ++num_decision_histories_;

  assert(decision_point.OpenSpielStatePtr());
  const std::vector<double> policy =
      profile_.Response(*decision_point.OpenSpielStatePtr());

  Cfv state_value = 0;
  if (decision_point.PlayerToAct() == player_) {
    sampler_.SampleTargetPlayerActions(
        policy,
        [this, &decision_point, importance_weighted_reach_prob, &state_value](
            int action_idx, double action_prob, double sampling_prob) {
          if (action_prob > 0) {
            state_value += (*this)(
                decision_point,
                action_prob * importance_weighted_reach_prob / sampling_prob,
                action_idx);
          }
        });
  } else {
    sampler_.SampleExternalPlayerActions(
        policy,
        [this, importance_weighted_reach_prob, &state_value, &decision_point](
            int action_idx, double action_prob, double sampling_prob) {
          if (action_prob > 0) {
            state_value += (*this)(
                decision_point,
                action_prob * importance_weighted_reach_prob / sampling_prob,
                action_idx);
          }
        });
  }
  return state_value;
}

// Evaluate the player's expected return and decision-history count under the given profile and sampler.
std::pair<Cfv, int> PolicyValue(DecisionPoint& root, int player,
                                const Policy& profile, MccfrSampler& sampler) {
  PolicyValueEvaluator evaluator(player, profile, sampler, root.NumPlayers()); // Construct the policy-value evaluator.
  const Cfv ev = evaluator(root, 1.0, 0); // Evaluate the artificial root's action zero through operator().
  return {ev, evaluator.num_decision_histories_};
}

// Compute the expected value and counterfactual-value tree.
// Carry the importance-weighted reach probability through the traversal.
Cfv PolicyCfValueTreeEvaluator::ComputeCfValueTree(
    std::vector<std::string>& siblings, DecisionPoint& decision_point,
    const Policy& profile, MccfrSampler& sampler,
    double importance_weighted_reach_prob) {
  ++num_decision_histories_; // Count this decision history.

  assert(decision_point.OpenSpielStatePtr());
    // Obtain the action probabilities from the supplied policy profile.
  std::vector<double> policy;
  policy = profile.Response(*decision_point.OpenSpielStatePtr());

  // Accumulate the target player's expected return, weighting other players' actions by their probabilities.
  Cfv state_value = 0.0; 
  if (SaveRegrets(decision_point.PlayerToAct())) { // Store regrets at the designated regret player's decision nodes.
    // Initialize this information state's counterfactual-value node.
    // Create a CfValueTreeNode when this information-state key is first encountered.
    if (!cf_value_tree_.contains(decision_point.InformationStateStringRef())) { 
      cf_value_tree_.emplace(decision_point.InformationStateStringRef(), // Construct the node under its information-state key.
                             decision_point.NumActions()); // Initialize one counterfactual value per action.
      siblings.push_back(decision_point.InformationStateStringRef()); // Record this target-player information state among the siblings.
    }
    // Compute counterfactual action values.
    double action_values[decision_point.NumActions()]; // One counterfactual value per action.
    // double iwrp = 0.0; // Importance-weighted reach probability.
    sampler.SampleTargetPlayerActions( // Enumerate or sample the target player's actions.
        policy, [this, &decision_point, importance_weighted_reach_prob,
                 &action_values, &state_value, &sampler, &profile](
                    int action_idx, double action_prob, double sampling_prob) {
          std::vector<std::string> child_keys;
          const Cfv cfv = CounterfactualValue(
              child_keys, decision_point, profile, sampler,
              importance_weighted_reach_prob / sampling_prob, action_idx);
          action_values[action_idx] = cfv; // The action's counterfactual value is obtained from its descendants.
          state_value += action_prob * cfv; // Expected value is the probability-weighted sum of action values.
          // iwrp += action_prob * importance_weighted_reach_prob / sampling_prob;
          if (child_keys.size() > 0) { // Append any target-player information states reached through this action.
            // Store the successor information-state keys under the current action.
            Concat(cf_value_tree_.at(decision_point.InformationStateStringRef())
                       .child_keys_[action_idx],
                   child_keys);
          }
        });
    auto& cf_values =
        cf_value_tree_.at(decision_point.InformationStateStringRef())
            .cf_values_; // Counterfactual values for the current information state.
    // Accumulate importance-weighted reach probabilities over histories in this information state.
    cf_value_tree_.at(decision_point.InformationStateStringRef()).iwrp_ += 
        importance_weighted_reach_prob; 
    cf_values.ev_ += state_value; // Accumulate the expected value.
    // Debugging check for the counterfactual value at information state "2b".
    // if (decision_point.InformationStateString() == "2b") {
    //   SPIEL_CHECK_EQ(state_value, 0.0);
    // }
    assert(decision_point.NumActions() == cf_values.Size());
    // Accumulate each action's counterfactual value.
    for (size_t a = 0; a < decision_point.NumActions(); ++a) {
      cf_values.v_[a] += action_values[a];
    }
  } else { // Other players' nodes do not store regrets here.
    // Enumerate or sample the other player's actions.
    sampler.SampleExternalPlayerActions(
        policy, [this, importance_weighted_reach_prob, &state_value,
                 &decision_point, &siblings, &sampler, &profile](
                    int action_idx, double action_prob, double sampling_prob) {
          if (action_prob >= 0) { // Include actions with nonnegative probability.
            state_value += CounterfactualValue( // Weight descendant values by the other player's action and sampling probabilities.
                siblings, decision_point, profile, sampler,
                action_prob * importance_weighted_reach_prob / sampling_prob, 
                action_idx);
          }
        });
  }
  return state_value;
}

// Compute the counterfactual value of the selected action at a decision point.
// siblings collects sibling information-state keys with the same predecessor.
// Carry the policy, sampler, importance-weighted reach probability, and action index through recursion.
Cfv PolicyCfValueTreeEvaluator::CounterfactualValue(
    std::vector<std::string>& siblings, DecisionPoint& decision_point,
    const Policy& profile, MccfrSampler& sampler,
    double importance_weighted_reach_prob, int action_idx) {
  Cfv v = 0;
  // Enumerate or sample the chance outcomes of this action.
  sampler.SampleChanceOutcomes(
      decision_point.OutcomeProbabilities(action_idx),// Chance probabilities between this decision point and its successors.
      [this, action_idx, &decision_point, importance_weighted_reach_prob, &v,
       &siblings, &profile, &sampler](int outcome, double outcome_prob, 
                                      double outcome_sampling_prob) {
        decision_point.Apply(action_idx, outcome); // Move to the selected successor, caching its outcomes if needed.
        const double next_iwrp = importance_weighted_reach_prob * outcome_prob / // Update the importance-weighted chance reach probability.
                                 outcome_sampling_prob; // outcome_sampling_prob：1.0
        v += decision_point.IsTerminal()
                 ? Return(decision_point) * next_iwrp
                 : ComputeCfValueTree(siblings, decision_point, profile,
                                      sampler, next_iwrp);
        decision_point.Undo(); // Return to the parent index while retaining the cached history.
      });
  return v;
}

}  // namespace hr_edl
