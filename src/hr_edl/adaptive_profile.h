#ifndef HR_EDL_ADAPTIVE_PROFILE_H_
#define HR_EDL_ADAPTIVE_PROFILE_H_

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

// PBCE profile: evaluate realized policies and apply the local perturbation update.
class PerturbedCfTreeLearnerProfile : public AdaptiveProfile {
 public:
  PerturbedCfTreeLearnerProfile(std::vector<CfValueTreeLearnerPtr>&& learners, double epsilon)
      : learners_(std::move(learners)),
        evaluators_(NewEvaluators()),
        epsilon_(epsilon) {}

  Policy* Strategy(size_t player) const override final { return learners_[player].get(); }

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
    return avg;
  }

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
    double avg = 0.0;

    for (int player = 0; player < root.NumPlayers(); ++player) {
      auto [v, initial_info_states, cf_value_tree, _] =
          evaluators_[player].ComputeCfValueTreeEvaluation(root, PolicyRefProfile(learners_),
                                                           sampler);
      learners_[player]->Update(initial_info_states, *cf_value_tree, epsilon_);
      avg += (v - avg) / (player + 1.0);
    }

    return avg;
  }

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
  double epsilon_;
};

}  // namespace hr_edl

#endif  // HR_EDL_ADAPTIVE_PROFILE_H_
