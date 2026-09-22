#ifndef PBCE_LTBR_H_
#define PBCE_LTBR_H_

#include <functional>
#include <string>
#include <vector>

#include "hr_edl/adaptive_profile.h"
#include "hr_edl/tabular_learner.h"

// Use the newly accumulated regret without additional discounting.
inline double RmUpdate(double prev_regret, double next_regret, size_t iteration) {
  return next_regret;
}

// Regret matching assigns weight to the positive part of each regret.
inline void RmLink(const std::vector<double>& regrets,
                   const hr_edl::EnumerationConsumer<double>& yield) {
  for (size_t i = 0; i < regrets.size(); ++i) {
    yield(i, hr_edl::Relu(regrets[i]));
  }
}

struct LabeledAdaptiveProfile {
  std::string label_;
  const std::function<hr_edl::AdaptiveProfilePtr(size_t num_players,
                                                double utility_diameter)> New;
};

inline std::vector<std::string> LearnerProfileLabels(
    const std::vector<LabeledAdaptiveProfile>& algs) {
  std::vector<std::string> labels;
  labels.reserve(algs.size());
  for (const auto& a : algs) {
    labels.push_back(a.label_);
  }
  return labels;
}

inline std::vector<LabeledAdaptiveProfile> ltbr_perturbed_cfr_algs(
    double epsilon) {
  return {
      LabeledAdaptiveProfile{
          "A-EFR_IN",
          [epsilon](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::PerturbedCfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<
                    hr_edl::InformedActionSequencePredecessors>::NewList(num_players, RmUpdate,
                                                                         RmLink),
                epsilon));
          }},
      LabeledAdaptiveProfile{
          "CSPS-EFR",
          [epsilon](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(
                new hr_edl::PerturbedCfTreeLearnerProfile(
                    hr_edl::BehavioralDeviationTabularCfvLearner<
                        hr_edl::CausalPartialSequencePredecessors>::NewList(
                        num_players, RmUpdate, RmLink), epsilon));
          }},
      LabeledAdaptiveProfile{
          "CFPS-EFR",
          [epsilon](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::PerturbedCfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<
                    hr_edl::CounterfactualPartialSequencePredecessors>::NewList(num_players,
                                                                                RmUpdate, RmLink),
                epsilon));
          }},
      LabeledAdaptiveProfile{
          "CFR",
          [epsilon](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::PerturbedCfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<
                    hr_edl::ImmediateExternalSequencePredecessors>::NewList(
                    num_players, RmUpdate, RmLink),
                epsilon));
          }},
      LabeledAdaptiveProfile{
          "CFR_IN",
          [epsilon](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::PerturbedCfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<
                    hr_edl::ImmediateInternalSequencePredecessors>::NewList(
                    num_players, RmUpdate, RmLink),
                epsilon));
          }},
  };
}


#endif  // PBCE_LTBR_H_
