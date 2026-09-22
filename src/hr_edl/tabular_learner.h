#ifndef HR_EDL_TABULAR_LEARNER_H_
#define HR_EDL_TABULAR_LEARNER_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hr_edl/action_transformation.h"

namespace hr_edl {

struct StageExposureEntry {
  int player = -1;
  std::string info_state;
  bool is_external = false;
  size_t g = 0;
  double w = 0.0;
};

// Statistics whose accumulation window is exactly one adaptive-epsilon stage.
// Solver regrets are deliberately not represented here: they have lifetime
// semantics and are normally retained when a stage is changed.
struct StageRegretMetrics {
  double max_transformed_full_regret = 0.0;
  double max_original_full_regret = 0.0;
  double max_conditional_regret = 0.0;
  double sum_positive_conditional_regret = 0.0;
  double w_min = std::numeric_limits<double>::infinity();
  double w_min_positive = std::numeric_limits<double>::infinity();
  double w_sum = 0.0;
  double w_max = 0.0;
  double w_at_max_csr = 0.0;
  size_t num_w = 0;
  size_t num_positive_w = 0;
  size_t num_zero_or_tiny_w = 0;
  std::vector<StageExposureEntry> exposure_entries;
  bool has_bottleneck = false;
  int bottleneck_player = -1;
  std::string bottleneck_info_state;
  bool bottleneck_is_external = false;
  size_t bottleneck_g = 0;

  void AddW(double w, double numerical_eps) {
    w_min = std::min(w_min, w);
    w_max = std::max(w_max, w);
    w_sum += w;
    ++num_w;
    if (w > numerical_eps) {
      w_min_positive = std::min(w_min_positive, w);
      ++num_positive_w;
    } else {
      ++num_zero_or_tiny_w;
    }
  }

  void Merge(const StageRegretMetrics& other) {
    max_transformed_full_regret =
        std::max(max_transformed_full_regret, other.max_transformed_full_regret);
    max_original_full_regret =
        std::max(max_original_full_regret, other.max_original_full_regret);
    if (other.has_bottleneck &&
        (!has_bottleneck || other.max_conditional_regret > max_conditional_regret)) {
      max_conditional_regret = other.max_conditional_regret;
      w_at_max_csr = other.w_at_max_csr;
      has_bottleneck = true;
      bottleneck_player = other.bottleneck_player;
      bottleneck_info_state = other.bottleneck_info_state;
      bottleneck_is_external = other.bottleneck_is_external;
      bottleneck_g = other.bottleneck_g;
    }
    sum_positive_conditional_regret += other.sum_positive_conditional_regret;
    w_min = std::min(w_min, other.w_min);
    w_min_positive = std::min(w_min_positive, other.w_min_positive);
    w_sum += other.w_sum;
    w_max = std::max(w_max, other.w_max);
    num_w += other.num_w;
    num_positive_w += other.num_positive_w;
    num_zero_or_tiny_w += other.num_zero_or_tiny_w;
    exposure_entries.insert(exposure_entries.end(), other.exposure_entries.begin(),
                            other.exposure_entries.end());
  }

  void Finalize() {
    if (!std::isfinite(w_min)) w_min = 0.0;
    if (!std::isfinite(w_min_positive)) w_min_positive = 0.0;
  }

  double WMean() const { return num_w == 0 ? 0.0 : w_sum / num_w; }
  bool HasValidConditionalExposure() const { return num_zero_or_tiny_w == 0; }
};

namespace _tl {
// Interface for an immediate-response policy.
// Provides the action probability distribution at the current state.
class ImmediatePolicy {
 public:
  virtual ~ImmediatePolicy() = default;
  virtual std::vector<double> Response() const = 0;  // Return the policy vector.
  virtual void Response(double* out) const = 0;      // Write the policy to the buffer pointed to by out.
  virtual size_t NumActions() const = 0;
};

// CachedImmediatePolicy implements the ImmediatePolicy
// interface and caches a fixed policy for an information state.
class CachedImmediatePolicy : public ImmediatePolicy {
 public:
  // Construct from the number of actions or a policy vector.
  CachedImmediatePolicy(size_t num_actions) : policy_(num_actions, 1.0 / num_actions) {}
  CachedImmediatePolicy(std::vector<double>&& policy) : policy_(std::move(policy)) {}
  virtual ~CachedImmediatePolicy() = default;

  // Return the cached policy vector.
  std::vector<double> Response() const override final { return policy_; }
  // Write the policy to out; the caller must provide a sufficiently large buffer.
  void Response(double* out) const override final {
    assert(out);
    std::memcpy(out, policy_.data(), policy_.size() * sizeof(*out));
  }
  // Return the number of actions.
  size_t NumActions() const override final { return policy_.size(); }

 protected:
  std::vector<double> policy_;
};

// RegretTransformation is a function type that takes
// the previous regret, current regret, and iteration, and returns the updated regret.
using RegretTransformation =
    std::function<double(double prev_regret, double regret, size_t round_number)>;
// LinkFn takes a regret vector and a ReLU function.
// using EnumerationConsumer = std::function<void(int, T)>;
using LinkFn = std::function<void(const std::vector<double>&, const EnumerationConsumer<double>&)>;

// ReachProbLists stores the probability lists for the current node.
struct ReachProbLists {
  const std::vector<double> prev_;               // w^t: memory-state probabilities at iteration t.
  const std::vector<double> next_;               // w^{t+1}: memory-state probabilities at iteration t+1.
  const std::vector<size_t> remaining_queries_;  // Remaining query counts for the current node.
  size_t Size() const { return prev_.size(); }
};

// Immediate decision information for one information state; Update() updates its policy.
// Inherits CachedImmediatePolicy and stores policy_, internal/external transformations, and their regrets.
class ImmediateDecisionInfo : public CachedImmediatePolicy {
 public:
  // Construct from the numbers of actions, external predecessor probabilities, and internal predecessor probabilities.
  // Each node has internal and external transformations with corresponding predecessor probabilities, e.g. causal deviations.
  ImmediateDecisionInfo(size_t num_actions, size_t ex_num_pred_reach_probs,
                        size_t in_num_pred_reach_probs)
      : CachedImmediatePolicy(num_actions),       // Initialize the base class with a uniform policy.
        round_number_(1),                         // Start the iteration counter at 1.
        ex_phi_list_(ex_num_pred_reach_probs > 0  // Generate all external transformations if external predecessor probabilities exist.
                         ? SwapActionTranformation::External(num_actions)  // Generate all external transformations.
                         : std::vector<SwapActionTranformation>()),        // Otherwise, use an empty list.
        in_phi_list_(in_num_pred_reach_probs > 0  // Generate all internal transformations if internal predecessor probabilities exist.
                         ? SwapActionTranformation::Internal(num_actions)  // Generate all internal transformations.
                         : std::vector<SwapActionTranformation>()),        // Otherwise, use an empty list.
        ex_regrets_(ex_phi_list_.size() * ex_num_pred_reach_probs,
                    0),  // Initialize flattened external-transformation regrets to zero: number of transformations * number of memory
                         // states.
        in_regrets_(in_phi_list_.size() * in_num_pred_reach_probs, 0),
        org_ex_regrets_(ex_phi_list_.size() * ex_num_pred_reach_probs,
                        0),  // Initialize flattened original-space external-transformation regrets to zero.
        org_in_regrets_(in_phi_list_.size() * in_num_pred_reach_probs,
                        0),  // Initialize flattened original-space internal-transformation regrets to zero.
        stage_ex_regrets_(ex_phi_list_.size() * ex_num_pred_reach_probs, 0),
        stage_in_regrets_(in_phi_list_.size() * in_num_pred_reach_probs, 0),
        stage_org_ex_regrets_(ex_phi_list_.size() * ex_num_pred_reach_probs, 0),
        stage_org_in_regrets_(in_phi_list_.size() * in_num_pred_reach_probs, 0),
        // reached_ex_regrets_(ex_phi_list_.size() * ex_num_pred_reach_probs,
        // 0),  // Initialize flattened reached external-transformation regrets to zero.
        // reached_in_regrets_(in_phi_list_.size() * in_num_pred_reach_probs,
        // 0),  // Initialize flattened reached internal-transformation regrets to zero.
        reached_probs_sum_ex_(ex_num_pred_reach_probs,
                              0.0),  // Initialize flattened external-transformation reach-probability sums to zero.
        reached_probs_sum_in_(in_num_pred_reach_probs, 0.0),
        stage_reached_probs_sum_ex_(ex_num_pred_reach_probs, 0.0),
        stage_reached_probs_sum_in_(in_num_pred_reach_probs, 0.0),
        phi_sum_(num_actions)
  // cumulative_policy_(num_actions, 0.0)
  {}  // Initialize the cumulative policy to zero.
  virtual ~ImmediateDecisionInfo() = default;

  // Update decision information, compute regrets for the current policy, and update phi_sum_.
  // ex_pred_reach_prob_lists and in_pred_reach_prob_lists hold external and internal predecessor probabilities.
  // update_target updates regrets; f is the link function applied to regrets.
  void Update(const CfValues& cfvs, const RegretTransformation& update_target, const LinkFn& f,
              const ReachProbLists& ex_pred_reach_prob_lists,
              const ReachProbLists& in_pred_reach_prob_lists, double epsilon = 0.0,
              double iwrp = 0.0) {
    if (!std::isfinite(epsilon) || epsilon < 0.0) {
      throw std::invalid_argument("epsilon must be finite and non-negative");
    }
    // Update both lifetime diagnostics and exact, non-truncated stage-local
    // original-space accumulators before transforming the values.
    if (is_original_) {
      if (ex_pred_reach_prob_lists.Size() > 0) {
        UpdateUnperturbedPhiRegrets(org_ex_regrets_, reached_probs_sum_ex_,
                                    stage_org_ex_regrets_, stage_reached_probs_sum_ex_,
                                    ex_phi_list_, cfvs, update_target,
                                    ex_pred_reach_prob_lists, iwrp);
      }
      if (in_pred_reach_prob_lists.Size() > 0) {
        UpdateUnperturbedPhiRegrets(org_in_regrets_, reached_probs_sum_in_,
                                    stage_org_in_regrets_, stage_reached_probs_sum_in_,
                                    in_phi_list_, cfvs, update_target,
                                    in_pred_reach_prob_lists, iwrp);
      }
    }
    // Compute perturbed CFVs and map back to the original policy space.
    if (epsilon > 0.0) {
      size_t num_actions = policy_.size();
      if (num_actions * epsilon >= 1.0) {
        throw std::invalid_argument(
            "epsilon must satisfy epsilon < 1 / num_actions at every information set");
      }
      // Compute phi regrets in the original game.
      CfValues perturbed_cfvs = cfvs;  // Copy the CFVs.
      double p_value = 0;
      for (size_t aidx = 0; aidx < num_actions; ++aidx) {
        p_value += cfvs[aidx] * epsilon;
      }
      for (size_t aidx = 0; aidx < num_actions; ++aidx) {
        perturbed_cfvs[aidx] = (1 - num_actions * epsilon) * cfvs[aidx] + p_value;
      }
      // Map the policy into the transformed game.
      for (size_t a = 0; a < num_actions; ++a) {
        policy_[a] = (policy_[a] - epsilon) / (1.0 - num_actions * epsilon);
      }
      is_original_ = false;  // Mark the policy as being in the perturbed space.

      Update(perturbed_cfvs, update_target, f, ex_pred_reach_prob_lists, in_pred_reach_prob_lists,
             0.0, iwrp);  // Avoid applying the perturbation again in the recursive call.
      // Map the transformed-game policy back to the original policy space.
      // Compute the perturbation term x_epsilon.
      // double x_epsilon = 0;
      double x_epsilon = epsilon;  // Since sum_{a} pi(a) = 1,
      // for (size_t a = 0; a < num_actions; ++a) {
      //   // x_epsilon = epsilon * sum_{a} pi(a) = epsilon * 1 = epsilon.
      // For a general vector epsilon with action-specific epsilon[a], x_epsilon += policy_[a] *
      //   epsilon;
      // }
      // Update the policy to (1 - num_actions*epsilon)*pi(a) + x_epsilon.
      for (size_t a = 0; a < num_actions; ++a) {
        policy_[a] = (1.0 - num_actions * epsilon) * policy_[a] + x_epsilon;
      }
      is_original_ = true;  // Mark the policy as being in the original space.
      return;
    }

    if (ex_pred_reach_prob_lists.Size() > 0) {  // Update the external-transformation regret array y.
      UpdatePhiSetRegrets(ex_regrets_, stage_ex_regrets_, ex_phi_list_, cfvs,
                          update_target, f, ex_pred_reach_prob_lists);
    }
    if (in_pred_reach_prob_lists.Size() > 0) {  // Update the internal-transformation regret matrix.
      UpdatePhiSetRegrets(in_regrets_, stage_in_regrets_, in_phi_list_, cfvs,
                          update_target, f, in_pred_reach_prob_lists);
    }
    phi_sum_.FixedPoint(policy_);  // Compute the fixed-point policy of phi_sum_ and store it in policy_.

    phi_sum_.Reset();  // Reset the total phi weight and the weighted external/internal phi arrays and matrices.
    ++round_number_;
  }



  // Get phi regrets.
  // max_full_phi_regret_fus: maximum over ex_regrets_ and in_regrets_, used for policy updates.
  // max_full_phi_regret: maximum over ex_regrets_ and in_regrets_ under the current policy.
  // max_phi_regret: maximum over reached_ex_regrets_ and
  // reached_in_regrets_, conditional on reaching this node.
  // sum_phi_regret: sum over reached_ex_regrets_ and
  // reached_in_regrets_, conditional on reaching this node.
  std::tuple<double, double, double, double> PhiRegrets() const {
    double max_full_phi_regret_fus = 0.0;  // for update strategy
    double max_full_phi_regret = 0.0;
    double max_phi_regret = 0.0;
    double sum_phi_regret = 0.0;
    for (size_t i = 0; i < ex_regrets_.size(); ++i) {
      // Compute external-transformation regrets.
      const double rs = ex_regrets_[i];
      const double r = org_ex_regrets_[i];
      const double w = reached_probs_sum_ex_[i % reached_probs_sum_ex_.size()];
      const double rr = w > 0.0 ? org_ex_regrets_[i] / w : 0.0;
      if (rs > max_full_phi_regret_fus) {
        max_full_phi_regret_fus = rs;
      }
      if (r > max_full_phi_regret) {
        max_full_phi_regret = r;
      }
      if (rr > max_phi_regret) {
        max_phi_regret = rr;
      }
      sum_phi_regret += std::max(0.0, rr);
    }
    // Compute internal-transformation regrets.
    for (size_t i = 0; i < in_regrets_.size(); ++i) {
      const double rs = in_regrets_[i];
      const double r = org_in_regrets_[i];
      const double w = reached_probs_sum_in_[i % reached_probs_sum_in_.size()];
      const double rr = w > 0.0 ? org_in_regrets_[i] / w : 0.0;
      if (rs > max_full_phi_regret_fus) {
        max_full_phi_regret_fus = rs;
      }
      if (r > max_full_phi_regret) {
        max_full_phi_regret = r;
      }
      if (rr > max_phi_regret) {
        max_phi_regret = rr;
      }
      sum_phi_regret += std::max(0.0, rr);
    }
    return {max_full_phi_regret_fus, max_full_phi_regret, max_phi_regret, sum_phi_regret};
  }

  StageRegretMetrics StageMetrics(double w_numerical_eps) const {
    StageRegretMetrics metrics;
    for (double regret : stage_ex_regrets_) {
      metrics.max_transformed_full_regret =
          std::max(metrics.max_transformed_full_regret, regret);
    }
    for (double regret : stage_in_regrets_) {
      metrics.max_transformed_full_regret =
          std::max(metrics.max_transformed_full_regret, regret);
    }

    const auto add_original_set =
        [&metrics, w_numerical_eps](const std::vector<double>& regrets,
                                    const std::vector<double>& reached_probs,
                                    bool is_external) {
          for (size_t g = 0; g < reached_probs.size(); ++g) {
            metrics.AddW(reached_probs[g], w_numerical_eps);
            metrics.exposure_entries.push_back(
                {-1, "", is_external, g, reached_probs[g]});
          }
          if (reached_probs.empty()) return;
          for (size_t i = 0; i < regrets.size(); ++i) {
            const double regret = regrets[i];
            metrics.max_original_full_regret =
                std::max(metrics.max_original_full_regret, regret);
            const size_t g = i % reached_probs.size();
            const double w = reached_probs[g];
            if (w <= w_numerical_eps) continue;
            const double conditional_regret = regret / w;
            if (conditional_regret > metrics.max_conditional_regret) {
              metrics.max_conditional_regret = conditional_regret;
              metrics.w_at_max_csr = w;
              metrics.has_bottleneck = true;
              metrics.bottleneck_is_external = is_external;
              metrics.bottleneck_g = g;
            } else if (!metrics.has_bottleneck) {
              metrics.w_at_max_csr = w;
              metrics.has_bottleneck = true;
              metrics.bottleneck_is_external = is_external;
              metrics.bottleneck_g = g;
            }
            metrics.sum_positive_conditional_regret +=
                std::max(0.0, conditional_regret);
          }
        };
    add_original_set(stage_org_ex_regrets_, stage_reached_probs_sum_ex_, true);
    add_original_set(stage_org_in_regrets_, stage_reached_probs_sum_in_, false);
    metrics.Finalize();
    return metrics;
  }

  void ResetStageEvaluation() {
    std::fill(stage_ex_regrets_.begin(), stage_ex_regrets_.end(), 0.0);
    std::fill(stage_in_regrets_.begin(), stage_in_regrets_.end(), 0.0);
    std::fill(stage_org_ex_regrets_.begin(), stage_org_ex_regrets_.end(), 0.0);
    std::fill(stage_org_in_regrets_.begin(), stage_org_in_regrets_.end(), 0.0);
    std::fill(stage_reached_probs_sum_ex_.begin(), stage_reached_probs_sum_ex_.end(), 0.0);
    std::fill(stage_reached_probs_sum_in_.begin(), stage_reached_probs_sum_in_.end(), 0.0);
  }

  void ResetSolverRegrets() {
    std::fill(ex_regrets_.begin(), ex_regrets_.end(), 0.0);
    std::fill(in_regrets_.begin(), in_regrets_.end(), 0.0);
    phi_sum_.Reset();
    round_number_ = 1;
  }

  void SetCfrPlus(bool enabled) { is_cfr_plus_ = enabled; }

  double SolverRegretL1() const {
    double total = 0.0;
    for (double r : ex_regrets_) total += std::abs(r);
    for (double r : in_regrets_) total += std::abs(r);
    return total;
  }

  double StageOriginalRegretL1() const {
    double total = 0.0;
    for (double r : stage_org_ex_regrets_) total += std::abs(r);
    for (double r : stage_org_in_regrets_) total += std::abs(r);
    return total;
  }

  double StageExposureSum() const {
    double total = 0.0;
    for (double w : stage_reached_probs_sum_ex_) total += w;
    for (double w : stage_reached_probs_sum_in_) total += w;
    return total;
  }

 private:
  // Update the immediate regrets of the phi set and add their weighted values to cumulative regrets.
  void UpdatePhiSetRegrets(std::vector<double>& regrets,
                           std::vector<double>& stage_regrets,
                           const std::vector<SwapActionTranformation>& phi_list,
                           const CfValues& cfvs, const RegretTransformation& update_target,
                           const LinkFn& f, const ReachProbLists& pred_reach_prob_lists) {
    const size_t num_reach_probs = pred_reach_prob_lists.prev_.size();
    assert(num_reach_probs ==
           pred_reach_prob_lists.next_.size());  // Check that the w^t and w^{t+1} probability lists have equal sizes.
    size_t idx = 0;
    // regrets[idx]:  [phi_1 with w_1, phi_1 with w_2,...,phi_1 with w_m, phi_2
    // with w_1,...,phi_2 with w_m,...]
    for (size_t i = 0; i < phi_list.size(); ++i) {
      const double regret = phi_list[i].Regret(cfvs, policy_);  // Compute this transformation's CFV regret: rho^{t}.
      for (size_t j = 0; j < num_reach_probs; ++j) {
        assert(idx < regrets.size());
        // Compute weighted regret w^{t}*rho^{t} and add it to the persistent cumulative regrets.
        // ltbr:
        // update_target currently returns its second argument, regret, but can be extended, e.g. for discounting.
        // regret
        const double instantaneous_regret = pred_reach_prob_lists.prev_[j] * regret;
        regrets[idx] += update_target(regrets[idx], instantaneous_regret, round_number_);
        stage_regrets[idx] += instantaneous_regret;
        // NOTE: CFR+ with ReLU is more stable, but eliminates the effect of k in QBCE.
        if (is_cfr_plus_ && regrets[idx] < 0.0) {
          regrets[idx] = 0.0;
        }
        ++idx;
      }
    }
    // Index of the current predecessor probability w.
    size_t reach_prob_idx = 0;
    // Index of the current phi.
    size_t phi_idx = 0;
    // Accumulate each phi's weighted regret w^{t+1} * (rho^{phi})^+.
    double sum = 0;
    // Update weighted_external_blocks_ or weighted_phi_blocks_ for external or internal phi.
    // Compute each phi's link value: y^{t+1}_{phi} = sum_{i} w^{t+1}_{i} *
    // (rho(phi))^+; f is the ReLU link function applied to each phi.
    f(regrets, [this, num_reach_probs, &reach_prob_idx, p_next = pred_reach_prob_lists.next_.data(),
                &sum, &phi_idx, &phi_list](size_t regret_idx, double link_output) {
      // y^{t+1}_{phi} += w^{t+1}_{phi} * (rho(phi))^+
      sum += p_next[reach_prob_idx] * link_output;
      ++reach_prob_idx;
      if (reach_prob_idx == num_reach_probs) {  // After visiting all predecessor probabilities for one phi,
        assert(phi_idx < phi_list.size());
        // add the external or internal phi and its total weight (memory-state
        // probability) to phi_sum_.
        phi_sum_.Add(phi_list[phi_idx],
                     sum);  // Add the transformation and weighted sum to phi_sum_, updating weighted_external_blocks_.
                            // or weighted_phi_blocks_
        ++phi_idx;
        sum = 0;
        reach_prob_idx = 0;
      }
    });
  }

  void UpdateUnperturbedPhiRegrets(std::vector<double>& regrets,
                                   std::vector<double>& reached_probs_sum,
                                   std::vector<double>& stage_regrets,
                                   std::vector<double>& stage_reached_probs_sum,
                                   const std::vector<SwapActionTranformation>& phi_list,
                                   const CfValues& cfvs, const RegretTransformation& update_target,
                                   const ReachProbLists& pred_reach_prob_lists,
                                   const double iwrp = 0.0) {
    const size_t num_reach_probs = pred_reach_prob_lists.prev_.size();
    assert(num_reach_probs ==
           pred_reach_prob_lists.next_.size());  // Check that the w^t and w^{t+1} probability lists have equal sizes.
    size_t idx = 0;
    // regrets[idx]:  [phi_1 with w_1, phi_1 with w_2,...,phi_1 with w_m, phi_2
    // with w_1,...,phi_2 with w_m,...]
    // Update cumulative reach probabilities.
    for (size_t j = 0; j < num_reach_probs; ++j) {
      const double exposure = pred_reach_prob_lists.prev_[j] * iwrp;
      reached_probs_sum[j] += exposure;
      stage_reached_probs_sum[j] += exposure;
    }
    for (size_t i = 0; i < phi_list.size(); ++i) {
      const double regret = phi_list[i].Regret(cfvs, policy_);  // Compute this transformation's CFV regret: rho^{t}.
      for (size_t j = 0; j < num_reach_probs; ++j) {
        assert(idx < regrets.size());
        // Compute weighted regret w^{t}*rho^{t} and add it to the persistent cumulative regrets.
        // ltbr:
        // update_target currently returns its second argument, regret, but can be extended, e.g. for discounting.
        // regret
        const double instantaneous_regret = pred_reach_prob_lists.prev_[j] * regret;
        regrets[idx] += update_target(regrets[idx], instantaneous_regret, round_number_);
        // Stage evaluation is the exact sum of instantaneous original-space
        // regrets; it intentionally does not inherit CFR+ truncation.
        stage_regrets[idx] += instantaneous_regret;
        // Persist reached_regrets, the regrets conditional on reaching this node.
        // reached_regrets[idx] += update_target(
        // reached_regrets[idx], pred_reach_prob_lists.prev_[j] * regret / iwrp, round_number_);
        // NOTE: CFR+ with ReLU is more stable, but eliminates the effect of k in QBCE.
        if (is_cfr_plus_) {
          if (regrets[idx] < 0.0) {
            regrets[idx] = 0.0;
          }
          // if (reached_regrets[idx] < 0.0) {
          //   reached_regrets[idx] = 0.0;
          // }
        }
        ++idx;
      }
    }
  }

 private:
  size_t round_number_;                                     // Current iteration.
  const std::vector<SwapActionTranformation> ex_phi_list_;  // External transformations.
  const std::vector<SwapActionTranformation> in_phi_list_;  // Internal transformations.
  std::vector<double> ex_regrets_;                          // Cumulative external-transformation regrets.
  std::vector<double> in_regrets_;                          // Cumulative internal-transformation regrets.
  // Cumulative regrets in the original space.
  std::vector<double> org_ex_regrets_;  // Cumulative external regrets in the original space.
  std::vector<double> org_in_regrets_;  // Cumulative internal regrets in the original space.
  std::vector<double> stage_ex_regrets_;
  std::vector<double> stage_in_regrets_;
  std::vector<double> stage_org_ex_regrets_;
  std::vector<double> stage_org_in_regrets_;
  // Cumulative regrets conditional on reaching this node.
  // std::vector<double> reached_ex_regrets_;
  // std::vector<double> reached_in_regrets_;
  std::vector<double> reached_probs_sum_ex_;  // Cumulative external reach probabilities for this node.
  std::vector<double> reached_probs_sum_in_;  // Cumulative internal reach probabilities for this node.
  std::vector<double> stage_reached_probs_sum_ex_;
  std::vector<double> stage_reached_probs_sum_in_;
  WeightedActionTransformation phi_sum_;      // Weighted action transformations used to update the policy.
  // Cumulative policy used to compute the average policy.
  //  std::vector<double> cumulative_policy_;
  // std::vector<double> perturbed_policy_;  // Current perturbed policy.
  bool is_original_ = true;  // Whether the policy is in the original space.
  bool is_cfr_plus_ = true;  // Whether cumulative solver/lifetime regrets use CFR+.
};
}  // namespace _tl

// Define predecessor and successor reach-probability computations for each deviation type: memory-state probabilities w^t,
// w^{t+1}
struct NoExternal {
  std::vector<double> ExternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    return {};
  }
};
struct AllExternal {
  std::vector<double> ExternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    return pred_reach_probs;
  }
};
struct NoInternal {
  std::vector<double> InternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    return {};
  }
};
struct AllInternal {
  std::vector<double> InternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    return pred_reach_probs;
  }
};
// Set counterfactual successor reach probabilities to 1.
struct CounterfactualSuccessors {
  std::pair<std::vector<double>, std::vector<size_t>> SuccessorReachProbs(
      const std::vector<double>& pred_reach_probs, const double* strat, size_t num_actions,
      size_t action, const std::vector<size_t>& remaining_queries) const {
    return {{1.0}, {}};
  }
};
// CFR_In: informed CF deviation
struct ImmediateInternalSequencePredecessors : public NoExternal,
                                               public AllInternal,
                                               public CounterfactualSuccessors {};

// CFR: blind CF deviation (original CFR):
// Use all external transformations, no internal transformations, and identity transformations at successors.
struct ImmediateExternalSequencePredecessors : public AllExternal,
                                               public NoInternal,
                                               public CounterfactualSuccessors {};

// CFR_Ex+In: a variant for testing the effect of Ex+In.
struct ImmediateExInSequencePredecessors : public AllExternal,
                                           public AllInternal,
                                           public CounterfactualSuccessors {};

// For identity transformations, successor reach probability is parent reach probability * current action probability. Note:
// there is only one reach probability.
struct IdentitySuccessors {
  std::pair<std::vector<double>, std::vector<size_t>> SuccessorReachProbs(
      const std::vector<double>& pred_reach_probs, const double* strat, size_t num_actions,
      size_t action, const std::vector<size_t>& remaining_queries) const {
    return {{pred_reach_probs[0] * strat[action]}, {}};
  }
};

// informed action deviation
// Each information set has one memory state, weighted by the player's reach probability under the current policy.
struct InformedActionSequencePredecessors : public NoExternal,
                                            public AllInternal,
                                            public IdentitySuccessors {};
// blind action deviation
struct BlindActionSequencePredecessors : public AllExternal,
                                         public NoInternal,
                                         public IdentitySuccessors {};

// Any predecessor can perform an external transformation followed by further external transformations. The current node's number of
// memory states equals the number of predecessors.
struct BlindPartialSequenceSuccessors {
  std::pair<std::vector<double>, std::vector<size_t>> SuccessorReachProbs(
      const std::vector<double>& pred_reach_probs, const double* strat, size_t num_actions,
      size_t action, const std::vector<size_t>& remaining_queries) const {
    std::vector<double> succ_reach_probs = pred_reach_probs;  // External
    succ_reach_probs.push_back(pred_reach_probs[pred_reach_probs.size() - 1] *
                               strat[action]);  // Identity
    return {succ_reach_probs, {}};
  }
};
// BPS: blind partial sequence deviation, single-target deviations generated
// form the set of blind causal deviations
struct BlindPartialSequencePredecessors : public AllExternal,
                                          public NoInternal,
                                          public BlindPartialSequenceSuccessors {};

// CFPS: counterfactual partial sequence deviation
// Reach the target information set through correlation, apply consecutive external transformations, then an internal transformation followed by recorrelation.
// The current node's number of memory states equals the number of predecessors.
struct CounterfactualPartialSequencePredecessors : public NoExternal,
                                                   public AllInternal,
                                                   public BlindPartialSequenceSuccessors {};

// CFPS_Ex+In: test the effect of Ex+In.
struct CounterfactualPartialSequenceExInPredecessors : public AllExternal,
                                                       public AllInternal,
                                                       public BlindPartialSequenceSuccessors {};

// Causal: compute successor memory-state probabilities, ordered as external, internal, and identity.
// The current node allows identity, internal, and external transformations; after a nonidentity transformation, successors allow only external transformations.
// ExternalPredecessorReachProbs/InternalPredecessorReachProbs below further filter out unneeded memory states.
// state
struct CausalSuccessors {
  std::pair<std::vector<double>, std::vector<size_t>> SuccessorReachProbs(
      const std::vector<double>& pred_reach_probs, const double* strat, size_t num_actions,
      size_t action, const std::vector<size_t>& remaining_queries) const {
    std::vector<double> succ_reach_probs = pred_reach_probs;  // External
    succ_reach_probs.reserve(pred_reach_probs.size() + num_actions);
    const double identity_seq =
        pred_reach_probs[pred_reach_probs.size() - 1];  // Place the identity probability last.
    // Internal.
    // For an internal transformation at the current node, retain probabilities of all nontarget actions; all predecessors still use identity transformations.
    for (size_t a = 0; a < num_actions; ++a) {
      if (a != action) {
        succ_reach_probs.push_back(identity_seq * strat[a]);
      }
    }
    succ_reach_probs.push_back(identity_seq *
                               strat[action]);  // Identity: place the probability of the current action last.
    return {succ_reach_probs, {}};
  }
};
// CSPS: Causal Partial Sequence Predecessors
// External transformations record memory-state probabilities that do not directly follow the recommendation at all predecessors: (n-1)*d cases.
// Internal transformations record the memory-state probability of directly following the recommendation: one case.
struct CausalPartialSequencePredecessors : public CausalSuccessors {
  // An external-transformation node can be reached through internal or external transformations; copy the parent's memory
  // states except the final identity-transformation entry.
  std::vector<double> ExternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    std::vector<double> p(pred_reach_probs.size() - 1);
    std::copy(pred_reach_probs.begin(), pred_reach_probs.end() - 1,
              p.begin());  // Copy all entries except the last.
    return p;
  }
  // An internal-transformation node is reached only through identity transformations; copy the last entry.
  std::vector<double> InternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    return {pred_reach_probs[pred_reach_probs.size() - 1]};
  }
};
// TICS: twice-informed causal sequence deviation
// TICS:
// Reach the target information set through identity transformations, force arrival at the final node with consecutive counterfactual external transformations, then apply an internal transformation.
// Thus, internal-transformation memory states cover all preceding cases, while external transformations are absent.
struct TwiceInformedPartialSequencePredecessors : public NoExternal,
                                                  public AllInternal,
                                                  public CausalSuccessors {};
// TICS-Ex+In is for testing only.
struct TwiceInformedPartialSequenceExInPredecessors : public AllExternal,
                                                      public AllInternal,
                                                      public CausalSuccessors {};
// Behavioral deviation, the most complex
struct BehavioralPredecessors : public NoExternal, public AllInternal {
  std::pair<std::vector<double>, std::vector<size_t>> SuccessorReachProbs(
      const std::vector<double>& pred_reach_probs, const double* strat, size_t num_actions,
      size_t action, const std::vector<size_t>& remaining_queries) const {
    std::vector<double> succ_reach_probs;
    succ_reach_probs.reserve(pred_reach_probs.size() * num_actions);  // Multiply.
    for (auto p : pred_reach_probs) {
      for (size_t a = 0; a < num_actions; ++a) {
        succ_reach_probs.push_back(p * strat[a]);  // Multiply the reach probability by the recommended action's probability.
      }
    }
    return {succ_reach_probs, {}};
  }
};

// default external
// k_queries: query budget; a query is available while budget remains.
// sequential: whether queries must be consecutive.
// causal: whether successors after a query can be reached only through external transformations.
// Use an overriding design.
// external
// internal
// Do not require maximal use of the query budget.
struct QueryPredecessors_ : public AllExternal {
  // Constructor initializing the parameters.
  explicit QueryPredecessors_(size_t query_budget, bool sequential, bool causal)
      : sequential_(sequential), causal_(causal), query_budget_(query_budget) {}

  // For query-enabled deviations, return reach probabilities of predecessors with queries remaining; the current node's number of
  // memory states equals the number of predecessors.
  std::vector<double> InternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    std::vector<double> pred_reach_probs_with_query;
    assert(pred_reach_probs.size() == pred_remain_queries.size());
    for (size_t i = 0; i < pred_reach_probs.size(); ++i) {
      if (pred_remain_queries[i] > 0) {  // Retain a predecessor's reach probability if its remaining query count is positive.
        pred_reach_probs_with_query.push_back(pred_reach_probs[i]);
      }
    }
    return pred_reach_probs_with_query;
  }

  // Compute successor reach probabilities and remaining query counts.
  // Compute successor reach probabilities and query counts.
  std::pair<std::vector<double>, std::vector<size_t>> SuccessorReachProbs(
      const std::vector<double>& pred_reach_probs, const double* strat, size_t num_actions,
      size_t action, const std::vector<size_t>& pred_remain_queries) const {
    assert(pred_reach_probs.size() == pred_remain_queries.size());
    // Do not query: select the no-query option.
    std::vector<double> succ_reach_probs = pred_reach_probs;        // External
    std::vector<size_t> succ_remain_queries = pred_remain_queries;  // Leave the query count unchanged.
    // k-query: constrain only the query count.
    if (!sequential_ && !causal_) {
      // Query: for memory states with positive remaining budget, multiply by the recommended action's probability and decrement the budget.
      for (size_t i = 0; i < pred_reach_probs.size(); ++i) {
        if (pred_remain_queries[i] > 0) {
          // Query and obtain the corresponding recommendations in turn.
          for (size_t a = 0; a < num_actions; ++a) {
            succ_reach_probs.push_back(pred_reach_probs[i] *
                                       strat[a]);  // Multiply the reach probability by the recommended action's probability.
            succ_remain_queries.push_back(pred_remain_queries[i] - 1);  // Decrement the remaining query count.
          }
        }
      }
      return {succ_reach_probs, succ_remain_queries};
    }

    // sequential: queries must be consecutive.
    if (sequential_ && !causal_) {
      // Reach without querying; clear the remaining budget when this interrupts consecutive queries.
      // Check whether the remaining count equals the initial query budget.
      for (size_t i = 0; i < pred_remain_queries.size(); ++i) {
        // A remaining count below the initial budget means that a predecessor has already queried.
        if (pred_remain_queries[i] < query_budget_) {
          succ_remain_queries[i] = 0;
        }
      }
      // Reach after querying: for g with positive budget, query and obtain the corresponding recommendation.
      for (size_t i = 0; i < pred_remain_queries.size(); ++i) {
        if (pred_remain_queries[i] > 0) {
          for (size_t a = 0; a < num_actions; ++a) {
            succ_reach_probs.push_back(pred_reach_probs[i] *
                                       strat[a]);  // Multiply the reach probability by the nonrecommended action's probability.
            succ_remain_queries.push_back(pred_remain_queries[i] - 1);  // Decrement the query count for a nonrecommended action.
          }
        }
      }
      return {succ_reach_probs, succ_remain_queries};
    }
    // causal
    if (!sequential_ && causal_) {
      // Reach after querying; clear the budget when deviating from the recommendation.
      for (size_t i = 0; i < pred_remain_queries.size(); ++i) {
        if (pred_remain_queries[i] > 0) {
          // Reach after querying and deviating from the recommendation.
          for (size_t a = 0; a < num_actions; ++a) {
            if (a != action) {
              succ_reach_probs.push_back(pred_reach_probs[i] *
                                         strat[a]);  // Multiply the reach probability by the nonrecommended action's probability.
              succ_remain_queries.push_back(0);      // Clear the query budget for nonrecommended actions.
            }
          }
          // Reach after querying and following the recommendation.
          succ_reach_probs.push_back(pred_reach_probs[i] *
                                     strat[action]);                  // Multiply the reach probability by the recommended action's probability.
          succ_remain_queries.push_back(pred_remain_queries[i] - 1);  // Decrement the remaining query count.
        }
      }
      return {succ_reach_probs, succ_remain_queries};
    }

    // sequential + causal
    if (sequential_ && causal_) {
      // Reach without querying; clear the remaining budget for g that has already queried.
      // Criterion: remaining query count is below the budget.
      for (size_t i = 0; i < pred_remain_queries.size(); ++i) {
        if (pred_remain_queries[i] <
            query_budget_) {  // A remaining count below the initial budget means that a predecessor has already queried.
          succ_remain_queries[i] = 0;
        }
      }
      // Reach after querying, satisfying both sequential and causal requirements.
      for (size_t i = 0; i < pred_remain_queries.size(); ++i) {
        if (pred_remain_queries[i] > 0) {
          // Reach after querying and deviating from the recommendation; clear the query budget.
          for (size_t a = 0; a < num_actions; ++a) {
            if (a != action) {
              succ_reach_probs.push_back(pred_reach_probs[i] *
                                         strat[a]);  // Multiply the reach probability by the nonrecommended action's probability.
              succ_remain_queries.push_back(0);      // Clear the query budget for nonrecommended actions.
            }
          }
          // Reach by following the recommendation; decrement the budget after querying.
          succ_reach_probs.push_back(pred_reach_probs[i] *
                                     strat[action]);                  // Multiply the reach probability by the recommended action's probability.
          succ_remain_queries.push_back(pred_remain_queries[i] - 1);  // Decrement the remaining query count.
        }
      }
      return {succ_reach_probs, succ_remain_queries};
    }

    return {pred_reach_probs, pred_remain_queries};
  }

  const size_t query_budget_;
  const bool sequential_;
  const bool causal_;
};

// Use the available queries as fully as possible.
// In EFR, successor nodes use recorrelation and do not account for future queries.
// Thus, this does not improve computational efficiency; use all queries available at the current node.
struct QueryPredecessors : QueryPredecessors_ {
  // Constructor initializing the parameters.
  explicit QueryPredecessors(size_t query_budget, bool sequential, bool causal)
      : QueryPredecessors_(query_budget, sequential, causal) {}

  // blind: return predecessor probabilities with zero queries remaining.
  std::vector<double> ExternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    std::vector<double> pred_reach_probs_with_max_query;
    assert(pred_reach_probs.size() == pred_remain_queries.size());
    // Check for maximal budget use: a predecessor whose remaining count equals the initial budget has never queried, so its reach probability can be used.
    for (size_t i = 0; i < pred_reach_probs.size(); ++i) {
      if (pred_remain_queries[i] ==
          0) {  // If a predecessor's remaining count equals the initial budget, it has never queried and its reach probability can be used.
        pred_reach_probs_with_max_query.push_back(pred_reach_probs[i]);
      }
    }
    return pred_reach_probs_with_max_query;
  }

  // For query-enabled deviations, return reach probabilities of predecessors with queries remaining; the current node's number of
  // memory states equals the number of predecessors.
  std::vector<double> InternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    std::vector<double> pred_reach_probs_with_max_query;
    assert(pred_reach_probs.size() == pred_remain_queries.size());
    // Handle each QB structure separately.
    // qb-EFR or qb-EFR_SEQ: return only predecessor reach probabilities with the minimum remaining query count.
    if (!causal_) {
      // Find the minimum remaining query count; use 1 if it is 0.
      size_t min_remain_query =
          *std::min_element(pred_remain_queries.begin(), pred_remain_queries.end());
      if (min_remain_query == 0) {
        min_remain_query = 1;
      }
      for (size_t i = 0; i < pred_reach_probs.size(); ++i) {
        if (pred_remain_queries[i] ==
            min_remain_query) {  // Use a predecessor's reach probability when its remaining query count equals the minimum.
          pred_reach_probs_with_max_query.push_back(pred_reach_probs[i]);
        }
      }
      return pred_reach_probs_with_max_query;
    }

    // qb-cs or qb-seq+cs:
    // Return only reach probabilities for predecessors with nonzero remaining budget; do not require predecessors to query and follow recommendations.
    // Selecting only the minimum would require predecessors to use queries maximally and reach the node by following recommendations.
    // TODO: Test whether selecting only the minimum improves efficiency.
    if (causal_) {
      for (size_t i = 0; i < pred_reach_probs.size(); ++i) {
        if (pred_remain_queries[i] > 0) {  // Retain a predecessor's reach probability if its remaining query count is positive.
          pred_reach_probs_with_max_query.push_back(pred_reach_probs[i]);
        }
      }
      return pred_reach_probs_with_max_query;
    }
  }
};

// Abstract Policy subclass providing learner construction and the Update policy-update interface.
class CfValueTreeLearner;
using CfValueTreeLearnerPtr = std::unique_ptr<CfValueTreeLearner>;
class CfValueTreeLearner : public virtual Policy {
 public:
  template <class Subclass, class... Args>
  static std::vector<CfValueTreeLearnerPtr>
  NewList(  // Create a vector containing one CfValueTreeLearnerPtr for each player.
      size_t num_players, Args&&... args) {
    std::vector<CfValueTreeLearnerPtr> learners;
    learners.reserve(num_players);  // Reserve space.
    for (size_t i = 0; i < num_players; ++i) {
      learners.emplace_back(new Subclass(std::forward<Args>(args)...));  // Perfect-forward the constructor arguments.
    }  // args... are the Subclass constructor arguments; new returns a pointer to the instance.
       // args: DeviationSequencePredecessors(), RmUpdate, RmLink
    return learners;
  }

 public:
  virtual ~CfValueTreeLearner() = default;
  // Policy-update interface.
  virtual void Update(const std::vector<std::string>& initial_info_states,
                      const InfoStateUvm<CfValueTreeNode>& cf_value_tree, double epsilon = 0.0) = 0;
  // Clone interface returning an owning pointer to an instance of the same class.
  virtual CfValueTreeLearnerPtr Clone() const = 0;
  // initial_info_states: information-state strings of the initial sibling nodes.
  // virtual std::tuple<double, double, double, double> LearnerReachedPhiRegret(
  //     const std::vector<std::string>& initial_info_states,
  //     const InfoStateUvm<CfValueTreeNode>& cf_value_tree) const {
  //   return {0.0, 0.0, 0.0, 0.0};
  // }
  virtual std::tuple<double, double, double, double> PhiRegrets() const { return {0.0, 0.0, 0.0, 0.0}; }
  virtual StageRegretMetrics StageMetrics(double w_numerical_eps) const { return {}; }
  virtual void ResetStageEvaluation() {}
  virtual void ResetSolverRegrets() {}
  virtual void SetCfrPlus(bool enabled) {}
  virtual double SolverRegretL1() const { return 0.0; }
  virtual double StageOriginalRegretL1() const { return 0.0; }
  virtual double StageExposureSum() const { return 0.0; }
};

// TabularResponder implements the Policy interface.
// It uses InfoStateUvm<T> to store information states and their decision information.
template <class T>  // Decision-information type, e.g. ImmediateDecisionInfo.
class TabularResponder : public virtual Policy {
 public:
  TabularResponder() : local_info_map_() {}
  virtual ~TabularResponder() = default;

  // Implement Response from the Policy interface.
  // Return an action probability distribution over the current state's legal actions.
  std::vector<double> Response(const open_spiel::State& state) const override final {
    const std::string info_state = state.InformationStateString();  // Use the information-state string as the key.
    if (local_info_map_.contains(info_state)) {
      return local_info_map_.at(info_state).Response();
    } else {
      const int n = state.LegalActions().size();
      return std::vector<double>(n, 1.0 / n);
    }
  }

  // Policy overload for retrieving the average policy.
  std::vector<double> AverageResponse(const open_spiel::State& state) const {
    const std::string info_state = state.InformationStateString();  // Use the information-state string as the key.
    if (local_info_map_.contains(info_state)) {
      return local_info_map_.at(info_state).AveragePolicy();
    } else {
      const int n = state.LegalActions().size();
      return std::vector<double>(n, 1.0 / n);
    }
  }

 protected:
  InfoStateUvm<T> local_info_map_;  // Hash map with information-state strings as keys and decision information T as values.
                                    // Instantiated with ImmediateDecisionInfo.
};

// CFR-based behavioral-deviation learner defining the Update policy-update method.
template <class DeviationSequencePredecessors>
// Concept DeviationSequencePredecessors requires
// virtual std::vector<double> ExternalPredecessorReachProbs(
//     const std::vector<double>& pred_reach_probs) const;
// virtual std::vector<double> InternalPredecessorReachProbs(
//     const std::vector<double>& pred_reach_probs) const;
// virtual std::vector<double> SuccessorReachProbs(
//     const std::vector<double>& pred_reach_probs, const double* strat,
//     size_t num_actions, size_t action) const;
class BehavioralDeviationTabularCfvLearner
    : public CfValueTreeLearner,                             // Abstract class providing learner-list construction and
                                                             // the Update policy-update interface.
      public TabularResponder<_tl::ImmediateDecisionInfo> {  // Defines Response and stores information-state strings:
                                                             // decision_info
                                                             // information states and their decision information.
 public:
  template <class... Args>  // Variadic template arguments, with zero or more types deduced by the compiler.
  // Static member function callable through the class without constructing an instance.
  // Construct a list of pointers to BehavioralDeviationTabularCfvLearner with the specified template types.
  // Default behavior: pass update_target and f.
  static std::vector<CfValueTreeLearnerPtr> NewList(size_t num_players,
                                                    _tl::RegretTransformation update_target,
                                                    _tl::LinkFn f) {
    return CfValueTreeLearner::NewList<BehavioralDeviationTabularCfvLearner>(
        num_players, DeviationSequencePredecessors(), update_target, f);
  }

  // Overload for types such as QueryPredecessors, allowing k/seq/caus
  // to be forwarded to the predecessor-type constructor.
  static std::vector<CfValueTreeLearnerPtr> NewList(size_t num_players,
                                                    _tl::RegretTransformation update_target,
                                                    _tl::LinkFn f, size_t query_budget,
                                                    bool sequential = false, bool causal = false) {
    return CfValueTreeLearner::NewList<BehavioralDeviationTabularCfvLearner>(
        num_players, DeviationSequencePredecessors(query_budget, sequential, causal), update_target,
        f, query_budget, sequential, causal);
  }

 private:
  // Stack state storing an information-state string and its reach_prob_lists.
  struct StackState {
    const std::string* info_state_string_;         // Pointer to the information-state string.
    const _tl::ReachProbLists* reach_prob_lists_;  // Phi weights: predecessor probabilities when not deviating or when following recommendations.
    // const double own_reach_prob; //
    // The player's reach probability at this node, used to compute sequence-form policy probabilities.

    void Reset(const std::string* info_state_string, const _tl::ReachProbLists* reach_prob_lists) {
      info_state_string_ = info_state_string;
      reach_prob_lists_ = reach_prob_lists;
      // own_reach_prob = own_reach_prob;
    }
  };

 public:
  // Constructor.
  BehavioralDeviationTabularCfvLearner(DeviationSequencePredecessors dev_seq_predecessors,
                                       const _tl::RegretTransformation& update_target,
                                       const _tl::LinkFn& f, size_t query_budget = 0,
                                       bool sequential = false, bool causal = false)
      : TabularResponder(),  // Initialize TabularResponder with an empty local_info_map_.
        update_target_(update_target),
        f_(f),
        dev_seq_predecessors_(std::move(dev_seq_predecessors)),
        query_budget_(query_budget),
        sequential_(sequential),
        causal_(causal) {}
  virtual ~BehavioralDeviationTabularCfvLearner() = default;

  // Update the policy from initial information states and cf_value_tree in top-down depth-first order.
  void Update(const std::vector<std::string>&
                  initial_info_states,  // initial_info_states: information-state strings of the initial sibling nodes.
              const InfoStateUvm<CfValueTreeNode>& cf_value_tree,
              double epsilon = 0.0) override final {
    std::vector<StackState> state_stack(
        cf_value_tree.size());  // Stack of pointers to information-state strings and reach_prob_lists, initially null.
    const auto initial_reach_prob_lists = new _tl::ReachProbLists{
        {1.0}, {1.0}, {query_budget_}};  // Initialize constant predecessor probability lists, with prev_ and next_ both {1.0}.
    int stack_idx = 0;
    // Push all initial_info_states
    // onto the stack with their information-state strings and reach_prob_lists, using reach probability 1.
    for (; stack_idx < initial_info_states.size(); ++stack_idx) {
      state_stack[stack_idx].Reset(&(initial_info_states[stack_idx]), initial_reach_prob_lists);
    }
    --stack_idx;  // Point to the final root node at the top of the stack, undoing the extra increment.
    std::vector<_tl::ReachProbLists*> all_reach_prob_lists = {
        initial_reach_prob_lists};  // Initialize the vector of reach-probability-list pointers for later cleanup.
    all_reach_prob_lists.reserve(cf_value_tree.size());  // Reserve space to avoid repeated reallocations.
    // Begin policy updates.
    while (stack_idx >= 0) {
      const auto [info_state_string_ptr, pred_reach_probs] =
          state_stack[stack_idx];  // Get the information-state string and ReachProbLists at the top of the stack.
                                   // Own reach probability: should this use the policy before or after the update?
      --stack_idx;                 // Pop the stack.

      const auto& [cf_values, child_keys, iwrp] =
          cf_value_tree.at(*info_state_string_ptr);  // Get cf_values and child_keys for the current information state.
                                                     // iwrp
      const size_t num_actions = cf_values.Size();   // Number of actions at the current information state.

      // With only one action, skip the policy update, update child reach_prob_lists directly, and point stack_idx to the children.
      if (num_actions < 2) {
        for (size_t child_idx = 0; child_idx < child_keys[0].size(); ++child_idx) {
          ++stack_idx;  // Push sibling children onto the stack in order.
          state_stack[stack_idx].Reset(
              &(child_keys[0][child_idx]),  // Store the information-state string and the current node's reach_prob_lists.
              pred_reach_probs);
        }
        continue;
      }
      // Get external-transformation reach probabilities w^t and w^{t+1}.
      const _tl::ReachProbLists ex_reach_probs = {
          dev_seq_predecessors_.ExternalPredecessorReachProbs(pred_reach_probs->prev_,
                                                              pred_reach_probs->remaining_queries_),
          dev_seq_predecessors_.ExternalPredecessorReachProbs(
              pred_reach_probs->next_, pred_reach_probs->remaining_queries_)};
      // Get internal-transformation reach probabilities w^t and w^{t+1}.
      // For query deviations, internal transformations require a query and thus a positive remaining budget.
      const _tl::ReachProbLists in_reach_probs = {
          dev_seq_predecessors_.InternalPredecessorReachProbs(pred_reach_probs->prev_,
                                                              pred_reach_probs->remaining_queries_),
          dev_seq_predecessors_.InternalPredecessorReachProbs(
              pred_reach_probs->next_, pred_reach_probs->remaining_queries_)};

      // Debug: print external- and internal-transformation reach probabilities.
      if (false && t < 10 && infoset < 10) {
        ++infoset;
        std::cout << infoset << "| ex_reach_probs prev: ";
        for (const auto& p : ex_reach_probs.prev_) {
          std::cout << p << " ";
        }
        std::cout << "ex_reach_probs next: ";
        for (const auto& p : ex_reach_probs.next_) {
          std::cout << p << " ";
        }
        std::cout << "in_reach_probs prev: ";
        for (const auto& p : in_reach_probs.prev_) {
          std::cout << p << " ";
        }
        std::cout << "in_reach_probs next: ";
        for (const auto& p : in_reach_probs.next_) {
          std::cout << p << " ";
        }
        std::cout << std::endl;
      }
      // auto=ImmediateDecisionInfo
      // Get or create decision information for the current information state, including its policy and regrets.
      // This class inherits TabularResponder<_tl::ImmediateDecisionInfo>.
      // Create the player's decision node for policy updates.
      auto& local_decision_info =
          GetOrCreateFromArgs(local_info_map_, *info_state_string_ptr, num_actions,
                              ex_reach_probs.Size(), in_reach_probs.Size());
      local_decision_info.SetCfrPlus(is_cfr_plus_);

      double prev_policy[num_actions];            // Store the current policy.
      local_decision_info.Response(prev_policy);  // Get the current policy and store it in prev_policy.
      // Update this information state's policy by computing external and internal phi regrets.
      local_decision_info.Update(cf_values, update_target_, f_, ex_reach_probs, in_reach_probs,
                                 epsilon, iwrp);

      double next_policy[num_actions];            // Store the updated policy.
      local_decision_info.Response(next_policy);  // Get the updated policy and store it in next_policy.

      // Update child reach_prob_lists.
      for (size_t a = 0; a < num_actions; ++a) {
        if (child_keys[a].size() < 1) {
          continue;
        }
        // Compute reach-probability lists w^t and w^{t+1} for the player's successor nodes under each action a.
        // Query deviations need remaining query counts; nonquery deviations do not.
        // For query deviations, SuccessorReachProbs returns successor reach probabilities and remaining query counts for each predecessor; otherwise, it returns only reach probabilities.
        // _tl::ReachProbLists* successor_reach_prob_lists; //
        // Pointer to the reach_prob_lists created for each child.
        const auto [succ_reach_probs, succ_remain_queries] =
            dev_seq_predecessors_.SuccessorReachProbs(pred_reach_probs->prev_, prev_policy,
                                                      num_actions, a,
                                                      pred_reach_probs->remaining_queries_);
        const auto [succ_reach_probs_next, succ_remain_queries_next] =
            dev_seq_predecessors_.SuccessorReachProbs(pred_reach_probs->next_, next_policy,
                                                      num_actions, a,
                                                      pred_reach_probs->remaining_queries_);
        // Create new reach_prob_lists for each child,
        // containing reach probabilities and remaining query counts; query counts do not depend on the policy.
        const auto successor_reach_prob_lists =
            new _tl::ReachProbLists{succ_reach_probs, succ_reach_probs_next, succ_remain_queries};

        // Push all children with their information-state strings and reach_prob_lists, updating policies from root to leaves.
        for (size_t child_idx = 0; child_idx < child_keys[a].size(); ++child_idx) {
          ++stack_idx;
          state_stack[stack_idx].Reset(&(child_keys[a][child_idx]), successor_reach_prob_lists);
        }
        all_reach_prob_lists.push_back(successor_reach_prob_lists);
      }
    }
    // After the iteration, free all dynamically allocated reach_prob_lists.
    if (debug_print_) {
      // Debug: print the size of all_reach_prob_lists and the contents of each referenced list.
      for (size_t i = 0; i < all_reach_prob_lists.size(); ++i) {
        // std::cout << "all_reach_prob_lists size: " <<
        // all_reach_prob_lists.size() << std::endl;
        auto ptr = all_reach_prob_lists[i];
        std::cout << "reach_prob_lists :" << i << " content: " << std::endl;
        std::cout << "prev: ";
        for (const auto& p : ptr->prev_) {
          std::cout << p << " ";
        }
        std::cout << "next: ";
        for (const auto& p : ptr->next_) {
          std::cout << p << " ";
        }
        std::cout << "remaining_queries: ";
        for (const auto& q : ptr->remaining_queries_) {
          std::cout << q << " ";
        }
        std::cout << std::endl;
      }
      // debug_print_ = false;
    }
    for (auto ptr : all_reach_prob_lists) {
      delete ptr;
    }
    ++t;
    infoset = 0;
  }

  // Clone the current learner and return a pointer to the new instance.
  CfValueTreeLearnerPtr Clone() const override final {
    return CfValueTreeLearnerPtr(new BehavioralDeviationTabularCfvLearner(*this));  // *this
  }


  // Get maximum and total regrets over all information sets.
  std::tuple<double, double, double, double> PhiRegrets() const override final {
    double global_max_full_phi_regret_fus = 0.0;
    double global_max_full_phi_regret = 0.0;
    double global_max_phi_regret = 0.0;
    double global_sum_phi_regret = 0.0;
    for (const auto& [info_state_string, local_decision_info] : local_info_map_) {
      auto [max_full_phi_regret_fus, max_full_phi_regret, max_phi_regret, sum_phi_regret] = local_decision_info.PhiRegrets();
      if (max_full_phi_regret_fus > global_max_full_phi_regret_fus) {
        global_max_full_phi_regret_fus = max_full_phi_regret_fus;
      }
      if (max_full_phi_regret > global_max_full_phi_regret) {
        global_max_full_phi_regret = max_full_phi_regret;
      }
      if (max_phi_regret > global_max_phi_regret) {
        global_max_phi_regret = max_phi_regret;
      }
      global_sum_phi_regret += sum_phi_regret;
    }
    return {global_max_full_phi_regret_fus, global_max_full_phi_regret, global_max_phi_regret, global_sum_phi_regret};
  }

  StageRegretMetrics StageMetrics(double w_numerical_eps) const override final {
    StageRegretMetrics metrics;
    for (const auto& entry : local_info_map_) {
      auto local_metrics = entry.second.StageMetrics(w_numerical_eps);
      for (auto& exposure : local_metrics.exposure_entries) {
        exposure.info_state = entry.first;
      }
      if (local_metrics.has_bottleneck) {
        local_metrics.bottleneck_info_state = entry.first;
      }
      metrics.Merge(local_metrics);
    }
    metrics.Finalize();
    return metrics;
  }

  void ResetStageEvaluation() override final {
    for (auto& entry : local_info_map_) entry.second.ResetStageEvaluation();
  }

  void ResetSolverRegrets() override final {
    for (auto& entry : local_info_map_) entry.second.ResetSolverRegrets();
  }

  void SetCfrPlus(bool enabled) override final {
    is_cfr_plus_ = enabled;
    for (auto& entry : local_info_map_) entry.second.SetCfrPlus(enabled);
  }

  double SolverRegretL1() const override final {
    double total = 0.0;
    for (const auto& entry : local_info_map_) total += entry.second.SolverRegretL1();
    return total;
  }

  double StageOriginalRegretL1() const override final {
    double total = 0.0;
    for (const auto& entry : local_info_map_) total += entry.second.StageOriginalRegretL1();
    return total;
  }

  double StageExposureSum() const override final {
    double total = 0.0;
    for (const auto& entry : local_info_map_) total += entry.second.StageExposureSum();
    return total;
  }

 private:
  const _tl::RegretTransformation update_target_;  // Regret-update function.
  const _tl::LinkFn f_;                            // Regret link function.
  const DeviationSequencePredecessors
      dev_seq_predecessors_;  // Defines the computation of reach-probability memory states
                              // for transformation information sets and successor nodes.
  // double max_reached_phi_regret_ = 0.0; // Maximum phi regret.
  const size_t query_budget_;  // Query budget for query deviations.
  const bool sequential_;      // Whether updates are sequential.
  const bool causal_;          // Whether updates are causal.
  bool is_cfr_plus_ = true;
  // Debug print flag.
  bool debug_print_ = false;
  int t = 0;        // Debug iteration counter.
  int infoset = 0;  // Debug information-set counter.
};

}  // namespace hr_edl

#endif  // HR_EDL_TABULAR_LEARNER_H_
