#include "hr_edl/samplers.h"

#include "open_spiel/spiel_utils.h"

namespace hr_edl {

// Sample an action index from an epsilon mixture of the policy and uniform distribution.
// policy: action probabilities; random_number: a draw in [0, 1]; epsilon: exploration rate in [0, 1].
// Return the sampled action index.
int SampleActionIndex(const std::vector<double>& policy, double random_number, // The caller supplies a uniform random draw.
                      double epsilon) { // The declaration defaults epsilon to zero.
  double cumulative_prob = 0;
  int aidx = 0;
  for (; aidx < policy.size(); ++aidx) {
    const double prob =
        epsilon / policy.size() + (1.0 - epsilon) * policy[aidx];
    cumulative_prob += prob;
    if (cumulative_prob > random_number) {
      return aidx;
    }
  }
  return aidx - 1;
}

// Sample an action index.
// actions_and_probs: action-probability pairs; random_number: a uniform random draw.
// Return the sampled action index.
int SampleActionIndex(const open_spiel::ActionsAndProbs& actions_and_probs,
                      double random_number) {
  double cumulative_prob = 0;
  int a = 0;
  for (; a < actions_and_probs.size(); ++a) {
    cumulative_prob += actions_and_probs[a].second;
    if (cumulative_prob > random_number) {
      return a;
    }
  }
  return a - 1;
}

// Enumerate every chance outcome.
void SampleAllChanceOutcomes(
    const open_spiel::State& state,
    const std::function<void(const ActionAndProb&, double)>& f) {
  for (const auto outcome : state.ChanceOutcomes()) { // std::vector<std::pair<Action, double>>
    f(outcome, outcome.second);
  }
}

// Sample one chance outcome.
// random_number: uniform random draw; state: current state; f: callback.
// Invoke f once with the sampled outcome and its probability.
void SampleOneChanceOutcome(
    double random_number, const open_spiel::State& state,
    const std::function<void(const ActionAndProb&, double)>& f) {
  const auto outcomes = state.ChanceOutcomes();
  f(outcomes[SampleActionIndex(outcomes, random_number)], 1.0);
}

// Enumerate every action of the target player.
void SampleAllTargetPlayerActions(
    const std::vector<double>& policy,
    const std::function<void(int action_idx, double policy_prob,
                             double sampling_prob)>& f) {
  for (int action_idx = 0; action_idx < policy.size(); ++action_idx) {
    f(action_idx, policy[action_idx], 1.0);
  }
}

// Sample one target-player action using an epsilon-uniform exploration mixture.
void SampleOneTargetPlayerAction(
    double random_number, const std::vector<double>& policy,
    const std::function<void(int action_idx, double policy_prob,
                             double sampling_prob)>& f,
    double epsilon) {
  const int action_idx = SampleActionIndex(policy, random_number, epsilon); // Sample the action index from the exploration mixture.
  const double p = policy[action_idx]; // Probability under the original policy.
  f(action_idx, p, (1 - epsilon) * p + epsilon / policy.size()); // Report both policy probability and exploration-adjusted sampling probability.
}

// Enumerate every action of another player.
// policy: action probabilities; f: callback.
void SampleAllExternalPlayerActions(
    const std::vector<double>& policy,
    // policy_prob: policy probability; sampling_prob: sampling probability.
    const std::function<void(int action_idx, double policy_prob,
                             double sampling_prob)>& f) {
  for (int action_idx = 0; action_idx < policy.size(); ++action_idx) {
    f(action_idx, policy[action_idx], 1.0);
  }
}

// Sample one action of another player.
// random_number: uniform random draw; policy: action probabilities; f: callback.
void SampleOneExternalPlayerAction(
    double random_number, const std::vector<double>& policy,
    const std::function<void(int action_idx, double policy_prob,
                             double sampling_prob)>& f) {
  const int action_idx = SampleActionIndex(policy, random_number); // The default epsilon of zero samples directly from the policy.
  const double p = policy[action_idx];
  f(action_idx, p, p);
}

}  // namespace hr_edl
