#ifndef HR_EDL_TABULAR_LEARNER_H_
#define HR_EDL_TABULAR_LEARNER_H_

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
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

// Historical local-table statistics accumulated within one adaptive-epsilon stage.
// Field names remain stable for existing result files; they do not establish
// equivalence to full continuation regret in the current manuscript.
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
  size_t Size() const { return prev_.size(); }
};

// Invert x = epsilon * 1 + (1 - n * epsilon) * xi. Only round-off-sized
// violations are repaired; an infeasible cached execution policy is an error.
inline std::vector<double> IntendedPolicy(const std::vector<double>& realized,
                                          double epsilon) {
  if (!std::isfinite(epsilon) || epsilon < 0.0 || realized.empty()) {
    throw std::invalid_argument("expected a nonempty policy and finite non-negative epsilon");
  }
  const double scale = 1.0 - realized.size() * epsilon;
  if (scale <= 0.0) {
    throw std::invalid_argument(
        "epsilon must satisfy epsilon < 1 / num_actions at every information set");
  }
  const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
                           std::max<size_t>(1, realized.size());
  const double upper = epsilon + scale;
  double realized_sum = 0.0;
  double intended_sum = 0.0;
  std::vector<double> intended;
  intended.reserve(realized.size());
  for (double probability : realized) {
    if (!std::isfinite(probability) || probability < epsilon - tolerance ||
        probability > upper + tolerance) {
      throw std::invalid_argument("realized policy is outside the epsilon-constrained simplex");
    }
    realized_sum += probability;
    const double value = std::clamp((probability - epsilon) / scale, 0.0, 1.0);
    intended.push_back(value);
    intended_sum += value;
  }
  if (std::abs(realized_sum - 1.0) > tolerance || intended_sum <= 0.0) {
    throw std::invalid_argument("realized policy must sum to one");
  }
  // Preserve the unperturbed policy exactly, including its floating-point sum.
  if (epsilon == 0.0) return realized;
  for (double& probability : intended) probability /= intended_sum;
  return intended;
}

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
        reached_probs_sum_ex_(ex_num_pred_reach_probs,
                              0.0),  // Initialize flattened external-transformation reach-probability sums to zero.
        reached_probs_sum_in_(in_num_pred_reach_probs, 0.0),
        stage_reached_probs_sum_ex_(ex_num_pred_reach_probs, 0.0),
        stage_reached_probs_sum_in_(in_num_pred_reach_probs, 0.0),
        phi_sum_(num_actions) {}
  virtual ~ImmediateDecisionInfo() = default;

  // Update decision information, compute regrets for the current policy, and update phi_sum_.
  // ex_pred_reach_prob_lists and in_pred_reach_prob_lists hold external and internal predecessor probabilities.
  // update_target updates regrets; f is the link function applied to regrets.
  void Update(const CfValues& cfvs, const RegretTransformation& update_target, const LinkFn& f,
              const ReachProbLists& ex_pred_reach_prob_lists,
              const ReachProbLists& in_pred_reach_prob_lists, double epsilon = 0.0,
              double iwrp = 0.0) {
    Update(cfvs, update_target, f, ex_pred_reach_prob_lists, in_pred_reach_prob_lists,
           ex_pred_reach_prob_lists, in_pred_reach_prob_lists, epsilon, iwrp);
  }

  // Solver memory weights use xi; original-game regret and exposure diagnostics
  // use x. The overload above remains suitable when both channels coincide.
  void Update(const CfValues& cfvs, const RegretTransformation& update_target, const LinkFn& f,
              const ReachProbLists& ex_pred_reach_prob_lists,
              const ReachProbLists& in_pred_reach_prob_lists,
              const ReachProbLists& original_ex_pred_reach_prob_lists,
              const ReachProbLists& original_in_pred_reach_prob_lists,
              double epsilon = 0.0, double iwrp = 0.0) {
    const auto intended_policy = IntendedPolicy(policy_, epsilon);
    const auto compatible = [](const ReachProbLists& solver, const ReachProbLists& original) {
      return solver.prev_.size() == solver.next_.size() &&
             original.prev_.size() == original.next_.size() &&
             solver.Size() == original.Size();
    };
    if (!compatible(ex_pred_reach_prob_lists, original_ex_pred_reach_prob_lists) ||
        !compatible(in_pred_reach_prob_lists, original_in_pred_reach_prob_lists)) {
      throw std::invalid_argument("solver and diagnostic memory lists must have matching shapes");
    }
    // Update both lifetime diagnostics and exact, non-truncated stage-local
    // original-space accumulators before transforming the values.
    if (is_original_) {
      if (ex_pred_reach_prob_lists.Size() > 0) {
        UpdateUnperturbedPhiRegrets(org_ex_regrets_, reached_probs_sum_ex_,
                                    stage_org_ex_regrets_, stage_reached_probs_sum_ex_,
                                    ex_phi_list_, cfvs, update_target,
                                    original_ex_pred_reach_prob_lists, iwrp);
      }
      if (in_pred_reach_prob_lists.Size() > 0) {
        UpdateUnperturbedPhiRegrets(org_in_regrets_, reached_probs_sum_in_,
                                    stage_org_in_regrets_, stage_reached_probs_sum_in_,
                                    in_phi_list_, cfvs, update_target,
                                    original_in_pred_reach_prob_lists, iwrp);
      }
    }
    // Compute perturbed CFVs and map back to the original policy space.
    if (epsilon > 0.0) {
      size_t num_actions = policy_.size();
      // Apply the local action-realization channel to the counterfactual values.
      CfValues perturbed_cfvs = cfvs;  // Copy the CFVs.
      double p_value = 0;
      for (size_t aidx = 0; aidx < num_actions; ++aidx) {
        p_value += cfvs[aidx] * epsilon;
      }
      for (size_t aidx = 0; aidx < num_actions; ++aidx) {
        perturbed_cfvs[aidx] = (1 - num_actions * epsilon) * cfvs[aidx] + p_value;
      }
      // Recover the intended local policy from the cached realized policy.
      policy_ = intended_policy;
      is_original_ = false;  // Mark the policy as being in the perturbed space.

      Update(perturbed_cfvs, update_target, f, ex_pred_reach_prob_lists, in_pred_reach_prob_lists,
             0.0, iwrp);  // Avoid applying the perturbation again in the recursive call.
      // Map the updated intended local policy back to its realized policy.
      for (size_t a = 0; a < num_actions; ++a) {
        policy_[a] = (1.0 - num_actions * epsilon) * policy_[a] + epsilon;
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



  // Historical local-table diagnostics, retained for result compatibility.
  // These maxima are not full continuation-deviation regret evaluations.
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
        if (is_cfr_plus_ && regrets[idx] < 0.0) {
          regrets[idx] = 0.0;
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
  std::vector<double> reached_probs_sum_ex_;  // Cumulative external reach probabilities for this node.
  std::vector<double> reached_probs_sum_in_;  // Cumulative internal reach probabilities for this node.
  std::vector<double> stage_reached_probs_sum_ex_;
  std::vector<double> stage_reached_probs_sum_in_;
  WeightedActionTransformation phi_sum_;      // Weighted action transformations used to update the policy.
  bool is_original_ = true;  // Whether the policy is in the original space.
  bool is_cfr_plus_ = true;  // Whether cumulative solver/lifetime regrets use CFR+.
};
}  // namespace _tl

// The five deviation families exposed by this release. The same memory rule is
// evaluated separately on intended solver and realized diagnostic policies.
struct NoExternal {
  std::vector<double> ExternalPredecessorReachProbs(
      const std::vector<double>&) const { return {}; }
};
struct AllExternal {
  std::vector<double> ExternalPredecessorReachProbs(
      const std::vector<double>& probabilities) const { return probabilities; }
};
struct NoInternal {
  std::vector<double> InternalPredecessorReachProbs(
      const std::vector<double>&) const { return {}; }
};
struct AllInternal {
  std::vector<double> InternalPredecessorReachProbs(
      const std::vector<double>& probabilities) const { return probabilities; }
};

struct CounterfactualSuccessors {
  std::vector<double> SuccessorReachProbs(
      const std::vector<double>&, const double*, size_t, size_t) const {
    return {1.0};
  }
};
// CF-IN (historical output label CFR_IN).
struct ImmediateInternalSequencePredecessors
    : NoExternal, AllInternal, CounterfactualSuccessors {};
// CF (historical output label CFR).
struct ImmediateExternalSequencePredecessors
    : AllExternal, NoInternal, CounterfactualSuccessors {};

struct IdentitySuccessors {
  std::vector<double> SuccessorReachProbs(
      const std::vector<double>& probabilities, const double* strategy,
      size_t, size_t action) const {
    return {probabilities[0] * strategy[action]};
  }
};
// ACT-IN (historical output label A-EFR_IN).
struct InformedActionSequencePredecessors
    : NoExternal, AllInternal, IdentitySuccessors {};

struct BlindPartialSequenceSuccessors {
  std::vector<double> SuccessorReachProbs(
      const std::vector<double>& probabilities, const double* strategy,
      size_t, size_t action) const {
    std::vector<double> successors = probabilities;
    successors.push_back(probabilities.back() * strategy[action]);
    return successors;
  }
};
// CFPS: keep predecessor memories and append the identity continuation.
struct CounterfactualPartialSequencePredecessors
    : NoExternal, AllInternal, BlindPartialSequenceSuccessors {};

struct CausalSuccessors {
  std::vector<double> SuccessorReachProbs(
      const std::vector<double>& probabilities, const double* strategy,
      size_t num_actions, size_t action) const {
    std::vector<double> successors = probabilities;
    successors.reserve(probabilities.size() + num_actions);
    const double identity = probabilities.back();
    for (size_t a = 0; a < num_actions; ++a) {
      if (a != action) successors.push_back(identity * strategy[a]);
    }
    // The final entry always represents following the recommendation.
    successors.push_back(identity * strategy[action]);
    return successors;
  }
};
// CSPS: external transformations use nonidentity memories; internal
// transformations use the final identity memory.
struct CausalPartialSequencePredecessors : CausalSuccessors {
  std::vector<double> ExternalPredecessorReachProbs(
      const std::vector<double>& probabilities) const {
    return {probabilities.begin(), probabilities.end() - 1};
  }
  std::vector<double> InternalPredecessorReachProbs(
      const std::vector<double>& probabilities) const {
    return {probabilities.back()};
  }
};

// Compatibility-only families used by the archived historical experiments.
// Their old labels were CFR/CFR_IN, but they are not the paper's CF/CF-IN.
struct CounterfactualPartialSequenceExInPredecessors
    : AllExternal, AllInternal, BlindPartialSequenceSuccessors {};
struct TwiceInformedPartialSequencePredecessors
    : NoExternal, AllInternal, CausalSuccessors {};

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

 protected:
  InfoStateUvm<T> local_info_map_;  // Hash map with information-state strings as keys and decision information T as values.
                                    // Instantiated with ImmediateDecisionInfo.
};

// Tabular learner for the five PBCE release deviation families.
template <class DeviationSequencePredecessors>
class BehavioralDeviationTabularCfvLearner
    : public CfValueTreeLearner,
      public TabularResponder<_tl::ImmediateDecisionInfo> {
 public:
  static std::vector<CfValueTreeLearnerPtr> NewList(
      size_t num_players, _tl::RegretTransformation update_target, _tl::LinkFn f) {
    return CfValueTreeLearner::NewList<BehavioralDeviationTabularCfvLearner>(
        num_players, DeviationSequencePredecessors(), update_target, f);
  }

  BehavioralDeviationTabularCfvLearner(
      DeviationSequencePredecessors dev_seq_predecessors,
      const _tl::RegretTransformation& update_target, const _tl::LinkFn& f)
      : update_target_(update_target),
        f_(f),
        dev_seq_predecessors_(std::move(dev_seq_predecessors)) {}

  void Update(const std::vector<std::string>& initial_info_states,
              const InfoStateUvm<CfValueTreeNode>& cf_value_tree,
              double epsilon = 0.0) override final {
    if (!std::isfinite(epsilon) || epsilon < 0.0) {
      throw std::invalid_argument("epsilon must be finite and non-negative");
    }
    struct MemoryProbLists {
      _tl::ReachProbLists intended;
      _tl::ReachProbLists realized;
    };
    struct StackState {
      const std::string* info_state;
      const MemoryProbLists* probabilities;
    };
    std::vector<StackState> stack;
    stack.reserve(cf_value_tree.size());

    // Successor lists can be shared by sibling information states. Owning
    // pointers keep them stable and also release them if an update throws.
    std::vector<std::unique_ptr<MemoryProbLists>> probability_lists;
    probability_lists.reserve(cf_value_tree.size() + 1);
    probability_lists.push_back(
        std::make_unique<MemoryProbLists>(MemoryProbLists{{{1.0}, {1.0}}, {{1.0}, {1.0}}}));
    for (const auto& info_state : initial_info_states) {
      stack.push_back({&info_state, probability_lists.back().get()});
    }

    while (!stack.empty()) {
      const auto [info_state, predecessor] = stack.back();
      stack.pop_back();
      const auto& [cf_values, child_keys, iwrp] = cf_value_tree.at(*info_state);
      const size_t num_actions = cf_values.Size();

      if (num_actions < 2) {
        for (const auto& child : child_keys[0]) {
          stack.push_back({&child, predecessor});
        }
        continue;
      }

      const _tl::ReachProbLists external = {
          dev_seq_predecessors_.ExternalPredecessorReachProbs(predecessor->intended.prev_),
          dev_seq_predecessors_.ExternalPredecessorReachProbs(predecessor->intended.next_)};
      const _tl::ReachProbLists internal = {
          dev_seq_predecessors_.InternalPredecessorReachProbs(predecessor->intended.prev_),
          dev_seq_predecessors_.InternalPredecessorReachProbs(predecessor->intended.next_)};
      const _tl::ReachProbLists original_external = {
          dev_seq_predecessors_.ExternalPredecessorReachProbs(predecessor->realized.prev_),
          dev_seq_predecessors_.ExternalPredecessorReachProbs(predecessor->realized.next_)};
      const _tl::ReachProbLists original_internal = {
          dev_seq_predecessors_.InternalPredecessorReachProbs(predecessor->realized.prev_),
          dev_seq_predecessors_.InternalPredecessorReachProbs(predecessor->realized.next_)};
      auto& local = GetOrCreateFromArgs(local_info_map_, *info_state, num_actions,
                                       external.Size(), internal.Size());
      local.SetCfrPlus(is_cfr_plus_);
      const auto previous_policy = local.Response();
      const auto previous_intended = _tl::IntendedPolicy(previous_policy, epsilon);
      local.Update(cf_values, update_target_, f_, external, internal,
                   original_external, original_internal, epsilon, iwrp);
      const auto next_policy = local.Response();
      const auto next_intended = _tl::IntendedPolicy(next_policy, epsilon);

      // Carry both w^t and w^{t+1} in their own coordinates. Counterfactual
      // successor rules still return {1}, independently of either policy.
      for (size_t action = 0; action < num_actions; ++action) {
        if (child_keys[action].empty()) continue;
        auto previous = dev_seq_predecessors_.SuccessorReachProbs(
            predecessor->intended.prev_, previous_intended.data(), num_actions, action);
        auto next = dev_seq_predecessors_.SuccessorReachProbs(
            predecessor->intended.next_, next_intended.data(), num_actions, action);
        auto original_previous = dev_seq_predecessors_.SuccessorReachProbs(
            predecessor->realized.prev_, previous_policy.data(), num_actions, action);
        auto original_next = dev_seq_predecessors_.SuccessorReachProbs(
            predecessor->realized.next_, next_policy.data(), num_actions, action);
        probability_lists.push_back(std::make_unique<MemoryProbLists>(MemoryProbLists{
            {std::move(previous), std::move(next)},
            {std::move(original_previous), std::move(original_next)}}));
        const auto* successor = probability_lists.back().get();
        for (const auto& child : child_keys[action]) {
          stack.push_back({&child, successor});
        }
      }
    }
  }

  // Clone the current learner and return a pointer to the new instance.
  CfValueTreeLearnerPtr Clone() const override final {
    return CfValueTreeLearnerPtr(new BehavioralDeviationTabularCfvLearner(*this));  // *this
  }


  // Aggregate the historical local-table diagnostics over information sets.
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
  bool is_cfr_plus_ = true;
};

}  // namespace hr_edl

#endif  // HR_EDL_TABULAR_LEARNER_H_
