#include <cmath>
#include <iostream>

#include "bin/ltbr.h"
#include "hr_edl/decision_point.h"
#include "hr_edl/spiel_extra.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace {
void CheckPolicy(hr_edl::DecisionPoint& root, const hr_edl::PolicyProfile& policy,
                 double epsilon) {
  hr_edl::ForEachState(root, [&](const hr_edl::DecisionPoint& point) {
    if (point.PlayerToAct() < 0) return;
    const auto probabilities = policy.Response(*point.OpenSpielStatePtr());
    double sum = 0.0;
    for (double probability : probabilities) {
      SPIEL_CHECK_TRUE(std::isfinite(probability));
      SPIEL_CHECK_GE(probability, epsilon - 1e-12);
      sum += probability;
    }
    SPIEL_CHECK_FLOAT_NEAR(sum, 1.0, 1e-12);
  }, hr_edl::ALL_PLAYERS);
}

// Epsilon transitions must retain the execution policy and solver state while
// clearing exactly the stage diagnostics. Exercise every current and legacy
// family on a game with multiple information sets per player.
void CheckStageReset(bool legacy) {
  const auto game = open_spiel::LoadGame("kuhn_poker(players=3)");
  hr_edl::CachedDecisionPoint root(game->NewInitialState(), false, true);
  hr_edl::NullSampler sampler;
  const auto algorithms = ltbr_perturbed_cfr_algs(0.1, legacy);
  SPIEL_CHECK_EQ(algorithms.size(), 5);
  for (const auto& algorithm : algorithms) {
    auto learner = algorithm.New(game->NumPlayers(), 1.0);
    learner->SetCfrPlus(false);
    for (int iteration = 0; iteration < 8; ++iteration) {
      learner->UpdateAlternateAndReturnEv(root, sampler);
    }
    const auto before = learner->Frozen();
    CheckPolicy(root, before, 0.1);
    const double solver_regrets = learner->SolverRegretL1();
    SPIEL_CHECK_GT(solver_regrets, 0.0);
    SPIEL_CHECK_GT(learner->StageOriginalRegretL1(), 0.0);
    SPIEL_CHECK_GT(learner->StageExposureSum(), 0.0);
    learner->ShrinkEpsilon(0.05);
    learner->ResetStageEvaluation();
    SPIEL_CHECK_EQ(learner->Epsilon(), 0.05);
    SPIEL_CHECK_EQ(learner->SolverRegretL1(), solver_regrets);
    SPIEL_CHECK_EQ(learner->StageOriginalRegretL1(), 0.0);
    SPIEL_CHECK_EQ(learner->StageExposureSum(), 0.0);
    const auto after = learner->Frozen();
    hr_edl::ForEachState(root, [&](const hr_edl::DecisionPoint& point) {
      if (point.PlayerToAct() < 0) return;
      const auto previous = before.Response(*point.OpenSpielStatePtr());
      const auto current = after.Response(*point.OpenSpielStatePtr());
      SPIEL_CHECK_EQ(previous.size(), current.size());
      for (size_t action = 0; action < current.size(); ++action) {
        SPIEL_CHECK_EQ(previous[action], current[action]);
      }
    }, hr_edl::ALL_PLAYERS);
    const auto metrics = learner->StageMetrics(1e-12);
    SPIEL_CHECK_EQ(metrics.max_transformed_full_regret, 0.0);
    SPIEL_CHECK_EQ(metrics.max_original_full_regret, 0.0);
    SPIEL_CHECK_EQ(metrics.max_conditional_regret, 0.0);
    learner->UpdateAlternateAndReturnEv(root, sampler);
    CheckPolicy(root, learner->Frozen(), 0.05);
    SPIEL_CHECK_GT(learner->StageExposureSum(), 0.0);
  }
}
}  // namespace

int main() {
  CheckStageReset(false);
  CheckStageReset(true);
  std::cout << "PBCE core policy and stage-reset invariants passed." << std::endl;
}
