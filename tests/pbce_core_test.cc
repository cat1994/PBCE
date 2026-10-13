#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "bin/ltbr.h"
#include "hr_edl/decision_point.h"
#include "hr_edl/spiel_extra.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace {
constexpr double kTolerance = 1e-11;

void CheckVector(const std::vector<double>& actual,
                 const std::vector<double>& expected) {
  SPIEL_CHECK_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    SPIEL_CHECK_FLOAT_NEAR(actual[i], expected[i], kTolerance);
  }
}

void CheckMetrics(const hr_edl::StageRegretMetrics& actual,
                  const hr_edl::StageRegretMetrics& expected) {
  SPIEL_CHECK_FLOAT_NEAR(actual.max_transformed_full_regret,
                        expected.max_transformed_full_regret, kTolerance);
  SPIEL_CHECK_FLOAT_NEAR(actual.max_original_full_regret,
                        expected.max_original_full_regret, kTolerance);
  SPIEL_CHECK_FLOAT_NEAR(actual.max_conditional_regret,
                        expected.max_conditional_regret, kTolerance);
  SPIEL_CHECK_EQ(actual.exposure_entries.size(), expected.exposure_entries.size());
  for (size_t i = 0; i < actual.exposure_entries.size(); ++i) {
    SPIEL_CHECK_EQ(actual.exposure_entries[i].is_external,
                   expected.exposure_entries[i].is_external);
    SPIEL_CHECK_EQ(actual.exposure_entries[i].g, expected.exposure_entries[i].g);
    SPIEL_CHECK_FLOAT_NEAR(actual.exposure_entries[i].w,
                          expected.exposure_entries[i].w, kTolerance);
  }
}

// Only expose existing local public observations: the regression exercises the
// production tree traversal without making optimizer arrays writable to tests.
template <class Predecessors>
class InspectableLearner
    : public hr_edl::BehavioralDeviationTabularCfvLearner<Predecessors> {
 public:
  InspectableLearner()
      : hr_edl::BehavioralDeviationTabularCfvLearner<Predecessors>(
            Predecessors(), RmUpdate, RmLink) {}

  const hr_edl::_tl::ImmediateDecisionInfo& Local(const std::string& key) const {
    return this->local_info_map_.at(key);
  }
};

enum class MemoryFamily { kActIn, kCsps, kCfps, kCf, kCfIn };

// Independent reference recurrence for the binary synthetic chain below. Do
// not use the production SuccessorReachProbs or inverse-policy helper here:
// otherwise accidentally propagating x in both channels could pass this test.
std::vector<double> ReferenceSuccessors(MemoryFamily family,
                                        const std::vector<double>& memory,
                                        const std::vector<double>& policy,
                                        size_t action) {
  if (family == MemoryFamily::kCf || family == MemoryFamily::kCfIn) return {1.0};
  if (family == MemoryFamily::kActIn) return {memory[0] * policy[action]};
  auto next = memory;
  if (family == MemoryFamily::kCsps) {
    next.push_back(memory.back() * policy[1 - action]);
  }
  next.push_back(memory.back() * policy[action]);
  return next;
}

std::vector<double> ReferenceSelection(MemoryFamily family,
                                       const std::vector<double>& memory,
                                       bool external) {
  if (family == MemoryFamily::kCsps) {
    return external ? std::vector<double>(memory.begin(), memory.end() - 1)
                    : std::vector<double>{memory.back()};
  }
  const bool has_external = family == MemoryFamily::kCf;
  return external == has_external ? memory : std::vector<double>{};
}

std::vector<double> ReferenceIntended(const std::vector<double>& realized,
                                      double epsilon) {
  return {(realized[0] - epsilon) / (1.0 - 2.0 * epsilon),
          (realized[1] - epsilon) / (1.0 - 2.0 * epsilon)};
}

struct ReferenceLocalStatistics {
  std::vector<double> solver[2], transformed[2], original[2], exposure[2];

  void Accumulate(size_t set, const hr_edl::CfValues& values, double epsilon,
                  const std::vector<double>& solver_memory,
                  const std::vector<double>& original_memory, double reach) {
    const size_t memories = solver_memory.size();
    if (memories == 0) return;
    if (solver[set].empty()) {
      solver[set].assign(2 * memories, 0.0);
      transformed[set].assign(2 * memories, 0.0);
      original[set].assign(2 * memories, 0.0);
      exposure[set].assign(memories, 0.0);
    }
    // In the binary case each nonidentity internal map is also a constant
    // map, ordered target 1 then 0; external maps are ordered target 0 then 1.
    for (size_t phi = 0; phi < 2; ++phi) {
      const size_t target = set == 0 ? phi : 1 - phi;
      const double transformed_value = (1.0 - 2.0 * epsilon) * values[target]
                                        + epsilon * (values[0] + values[1]);
      for (size_t g = 0; g < memories; ++g) {
        const size_t index = phi * memories + g;
        const double increment = solver_memory[g] * (transformed_value - values());
        solver[set][index] = std::max(0.0, solver[set][index] + increment);
        transformed[set][index] += increment;
        original[set][index] += original_memory[g] * (values[target] - values());
      }
    }
    for (size_t g = 0; g < memories; ++g) exposure[set][g] += original_memory[g] * reach;
  }

  void Check(const hr_edl::_tl::ImmediateDecisionInfo& actual) const {
    double expected_solver_l1 = 0.0, max_transformed = 0.0, max_original = 0.0;
    double max_conditional = 0.0;
    const auto metrics = actual.StageMetrics(1e-12);
    size_t exposure_index = 0;
    for (size_t set = 0; set < 2; ++set) {
      for (size_t i = 0; i < solver[set].size(); ++i) {
        expected_solver_l1 += solver[set][i];
        max_transformed = std::max(max_transformed, transformed[set][i]);
        max_original = std::max(max_original, original[set][i]);
        const double w = exposure[set][i % exposure[set].size()];
        if (w > 1e-12) max_conditional = std::max(max_conditional, original[set][i] / w);
      }
      for (double w : exposure[set]) {
        SPIEL_CHECK_LT(exposure_index, metrics.exposure_entries.size());
        SPIEL_CHECK_FLOAT_NEAR(metrics.exposure_entries[exposure_index++].w, w, kTolerance);
      }
    }
    SPIEL_CHECK_EQ(exposure_index, metrics.exposure_entries.size());
    SPIEL_CHECK_FLOAT_NEAR(actual.SolverRegretL1(), expected_solver_l1, kTolerance);
    SPIEL_CHECK_FLOAT_NEAR(metrics.max_transformed_full_regret, max_transformed, kTolerance);
    SPIEL_CHECK_FLOAT_NEAR(metrics.max_original_full_regret, max_original, kTolerance);
    SPIEL_CHECK_FLOAT_NEAR(metrics.max_conditional_regret, max_conditional, kTolerance);
  }
};

// A root -> middle -> leaf chain tests products of multiple remembered
// recommendations, as well as the blind '*' entries that remain one. The
// controlled CFVs deliberately change sign so both previous-round regret
// weights and next-round fixed-point weights affect observable behavior.
template <class Predecessors>
void CheckMemoryChannels(MemoryFamily family, double epsilon) {
  InspectableLearner<Predecessors> learner;
  const std::array<std::string, 3> keys = {"root", "middle", "leaf"};
  const std::array<size_t, 2> path_actions = {0, 1};
  hr_edl::InfoStateUvm<hr_edl::CfValueTreeNode> tree;
  std::array<std::unique_ptr<hr_edl::_tl::ImmediateDecisionInfo>, 3> expected;
  std::array<ReferenceLocalStatistics, 3> statistics;
  std::vector<double> initial_memory = {1.0};
  for (size_t depth = 0; depth < keys.size(); ++depth) {
    tree.emplace(keys[depth], hr_edl::CfValueTreeNode(2));
    expected[depth] = std::make_unique<hr_edl::_tl::ImmediateDecisionInfo>(
        2, ReferenceSelection(family, initial_memory, true).size(),
        ReferenceSelection(family, initial_memory, false).size());
    if (depth < path_actions.size()) {
      tree.at(keys[depth]).child_keys_[path_actions[depth]].push_back(keys[depth + 1]);
      initial_memory = ReferenceSuccessors(family, initial_memory, {0.5, 0.5},
                                           path_actions[depth]);
    }
  }

  const std::array<std::array<double, 3>, 5> advantages = {{
      {{-2.0, 1.0, -3.0}}, {{1.0, -3.0, 2.0}}, {{3.0, 2.0, -1.0}},
      {{-1.0, -2.0, 3.0}}, {{2.0, 3.0, -2.0}}}};
  for (const auto& round_advantages : advantages) {
    for (size_t depth = 0; depth < keys.size(); ++depth) {
      const auto policy = expected[depth]->Response();
      auto& node = tree.at(keys[depth]);
      const double value = round_advantages[depth];
      node.cf_values_.v_ = {value, -value};
      node.cf_values_.ev_ = value * (policy[0] - policy[1]);
      node.iwrp_ = 0.9 - 0.2 * depth;
    }
    learner.Update({keys.front()}, tree, epsilon);

    std::vector<double> previous_solver = {1.0}, next_solver = {1.0};
    std::vector<double> previous_original = {1.0}, next_original = {1.0};
    for (size_t depth = 0; depth < keys.size(); ++depth) {
      if (epsilon == 0.0) {
        CheckVector(previous_solver, previous_original);
        CheckVector(next_solver, next_original);
      }
      if (family == MemoryFamily::kCf || family == MemoryFamily::kCfIn) {
        CheckVector(previous_solver, {1.0});
        CheckVector(next_solver, {1.0});
        CheckVector(previous_original, {1.0});
        CheckVector(next_original, {1.0});
      }
      const auto select = [&](const std::vector<double>& previous,
                              const std::vector<double>& next, bool external) {
        return hr_edl::_tl::ReachProbLists{
            ReferenceSelection(family, previous, external),
            ReferenceSelection(family, next, external)};
      };
      const auto previous_x = expected[depth]->Response();
      const auto& node = tree.at(keys[depth]);
      for (size_t set = 0; set < 2; ++set) {
        statistics[depth].Accumulate(
            set, node.cf_values_, epsilon,
            ReferenceSelection(family, previous_solver, set == 0),
            ReferenceSelection(family, previous_original, set == 0), node.iwrp_);
      }
      expected[depth]->Update(node.cf_values_, RmUpdate, RmLink,
                              select(previous_solver, next_solver, true),
                              select(previous_solver, next_solver, false),
                              select(previous_original, next_original, true),
                              select(previous_original, next_original, false),
                              epsilon, node.iwrp_);
      const auto next_x = expected[depth]->Response();
      const auto& actual = learner.Local(keys[depth]);
      statistics[depth].Check(actual);
      CheckVector(actual.Response(), next_x);
      CheckMetrics(actual.StageMetrics(1e-12), expected[depth]->StageMetrics(1e-12));
      SPIEL_CHECK_FLOAT_NEAR(actual.SolverRegretL1(),
                            expected[depth]->SolverRegretL1(), kTolerance);
      if (depth < path_actions.size()) {
        const auto action = path_actions[depth];
        previous_solver = ReferenceSuccessors(
            family, previous_solver, ReferenceIntended(previous_x, epsilon), action);
        next_solver = ReferenceSuccessors(
            family, next_solver, ReferenceIntended(next_x, epsilon), action);
        previous_original = ReferenceSuccessors(family, previous_original, previous_x, action);
        next_original = ReferenceSuccessors(family, next_original, next_x, action);
      }
    }
  }
}

// With a zero link the policy stays uniform, so the two rounds have exactly
// opposite regret vectors. Optimization must still use CFR+ clipping, while
// the evaluation path used by fixed and adaptive runs must report signed sums.
void CheckCfrPlusEvaluation(bool internal, double epsilon) {
  hr_edl::_tl::ImmediateDecisionInfo local(2, internal ? 0 : 1, internal ? 1 : 0);
  local.SetCfrPlus(true);
  const hr_edl::_tl::ReachProbLists one{{1.0}, {1.0}}, empty{{}, {}};
  const auto zero_link = [](const std::vector<double>& regrets,
                            const hr_edl::EnumerationConsumer<double>& yield) {
    for (size_t i = 0; i < regrets.size(); ++i) yield(i, 0.0);
  };
  for (double sign : {-1.0, 1.0}) {
    local.Update(hr_edl::CfValues(std::vector<double>{sign, -sign}, 0.0),
                 RmUpdate, zero_link, internal ? empty : one,
                 internal ? one : empty, epsilon, 1.0);
    CheckVector(local.Response(), {0.5, 0.5});
  }
  SPIEL_CHECK_FLOAT_NEAR(local.SolverRegretL1(), 1.0 - 2.0 * epsilon, kTolerance);
  const auto metrics = local.StageMetrics(1e-12);
  SPIEL_CHECK_FLOAT_NEAR(metrics.max_transformed_full_regret, 0.0, kTolerance);
  SPIEL_CHECK_FLOAT_NEAR(metrics.max_original_full_regret, 0.0, kTolerance);
  SPIEL_CHECK_FLOAT_NEAR(metrics.max_conditional_regret, 0.0, kTolerance);
  SPIEL_CHECK_FLOAT_NEAR(local.StageExposureSum(), 2.0, kTolerance);
  const auto old_state = local.SolverRegretL1();
  local.ResetStageEvaluation();
  SPIEL_CHECK_EQ(local.SolverRegretL1(), old_state);
  SPIEL_CHECK_EQ(local.StageExposureSum(), 0.0);
  local.Update(hr_edl::CfValues(std::vector<double>{1.0, -1.0}, 0.0),
               RmUpdate, zero_link, internal ? empty : one,
               internal ? one : empty, epsilon, 1.0);
  SPIEL_CHECK_FLOAT_NEAR(local.SolverRegretL1(), 2.0 * (1.0 - 2.0 * epsilon), kTolerance);
  SPIEL_CHECK_FLOAT_NEAR(local.StageMetrics(1e-12).max_transformed_full_regret,
                        1.0 - 2.0 * epsilon, kTolerance);
  SPIEL_CHECK_FLOAT_NEAR(local.StageMetrics(1e-12).max_original_full_regret, 1.0,
                        kTolerance);
}

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
  for (double epsilon : {0.0, 0.1}) {
    CheckMemoryChannels<hr_edl::InformedActionSequencePredecessors>(
        MemoryFamily::kActIn, epsilon);
    CheckMemoryChannels<hr_edl::CausalPartialSequencePredecessors>(
        MemoryFamily::kCsps, epsilon);
    CheckMemoryChannels<hr_edl::CounterfactualPartialSequencePredecessors>(
        MemoryFamily::kCfps, epsilon);
    CheckMemoryChannels<hr_edl::ImmediateExternalSequencePredecessors>(
        MemoryFamily::kCf, epsilon);
    CheckMemoryChannels<hr_edl::ImmediateInternalSequencePredecessors>(
        MemoryFamily::kCfIn, epsilon);
    CheckCfrPlusEvaluation(false, epsilon);
    CheckCfrPlusEvaluation(true, epsilon);
  }
  CheckStageReset(false);
  CheckStageReset(true);
  std::cout << "PBCE memory-channel, signed-evaluation, policy and stage-reset invariants passed."
            << std::endl;
}
