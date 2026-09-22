#ifndef HR_EDL_ACTION_TRANSFORMATION_H_
#define HR_EDL_ACTION_TRANSFORMATION_H_

#include <cassert>

#include "eigen/Eigen/Dense"
#include "hr_edl/math.h"
#include "hr_edl/types.h"

namespace hr_edl {

// Generate identity-matrix diagonal indices [0, 1, 2, ..., d-1].
inline std::vector<size_t> SquareIdentity(size_t d) {
  std::vector<size_t> eye;
  eye.reserve(d);
  for (size_t i = 0; i < d; ++i) {
    eye.push_back(i);
  }
  return eye;
}

// Check whether phi_block represents an external transformation.
// An external transformation maps every action to the same action.
inline bool RepresentsExternalTransformation(
    const std::vector<size_t>& phi_block) {
  const size_t ex_a = phi_block[0]; // Common target action index.
  for (size_t a = 1; a < phi_block.size(); ++a) {
    if (phi_block[a] != ex_a) {
      return false;
    }
  }
  return true;
}

// SwapActionTranformation handles action transformations.
// phi_block_ maps each action to its transformed action, ai -> aj.
// E.g. external -> a0: {a0, a0, a0, ...}; internal a1 -> a0: {a0, a0, a2, ...}.
// is_external_ indicates whether the transformation is external.
class SwapActionTranformation {
 public:
  // Generate all possible external transformations as SwapActionTranformation objects.
  static std::vector<SwapActionTranformation> External(size_t num_actions) {
    std::vector<SwapActionTranformation> list;
    list.reserve(num_actions);
    for (size_t a1 = 0; a1 < num_actions; ++a1) {
      list.emplace_back(std::vector<size_t>(num_actions, a1));
    }
    return list;
  }
  // Generate all possible internal transformations as SwapActionTranformation objects.
  static std::vector<SwapActionTranformation> Internal(size_t num_actions) {
    std::vector<SwapActionTranformation> list;
    list.reserve(num_actions * num_actions - num_actions); // n*n-n internal transformations.
    std::vector<size_t> phi_block = SquareIdentity(num_actions); // Equivalent to Python's range(d).
    for (size_t a1 = 0; a1 < num_actions; ++a1) {
      for (size_t a2 = 0; a2 < num_actions; ++a2) {
        if (a1 == a2) { // Exclude mappings of an action to itself.
          continue;
        }
        phi_block[a1] = a2; // Map action a1 to a2.
        list.emplace_back(phi_block); // Add a new transformation.
        phi_block[a1] = a1; // Restore the action-mapping vector.
      }
    }
    return list;
  }

 public:
  // Construct from an rvalue action-mapping vector.
  SwapActionTranformation(std::vector<size_t>&& phi_block)
      : phi_block_(std::move(phi_block)),
        is_external_(RepresentsExternalTransformation(phi_block_)) {}
  // Construct from a const-reference action-mapping vector.
  SwapActionTranformation(const std::vector<size_t>& phi_block)
      : phi_block_(phi_block),
        is_external_(RepresentsExternalTransformation(phi_block_)) {}
  virtual ~SwapActionTranformation() = default;

  // For a pure policy, map ai to aj and return a one-hot policy with probability 1 at aj.
  std::vector<double> operator()(size_t action) const {
    std::vector<double> phi_policy(NumActions(), 0); // Initialize a zero vector.
    phi_policy[SwapTarget(action)] = 1.0; // Set the target action's entry to 1.
    return phi_policy;
  }
  // For a mixed policy, construct the transformed policy using phi_block.
  std::vector<double> operator()(const std::vector<double>& policy) const {
    std::vector<double> phi_policy(policy.size(), 0);
    for (size_t action = 0; action < NumActions(); ++action) {
      phi_policy[SwapTarget(action)] += policy[action];
    }
    return phi_policy;
  }

  // Get the transformed action index for aidx.
  size_t SwapTarget(size_t action) const { return phi_block_[action]; }
  size_t NumActions() const { return phi_block_.size(); }

  // Compute the regret of this transformation under policy.
  double Regret(const CfValues& cfvs, const std::vector<double>& policy) const {
    double r = -cfvs(); // Negate the expected value using the overloaded operator.
    assert(policy.size() == cfvs.Size());
    assert(policy.size() == NumActions());
    for (size_t action = 0; action < policy.size(); ++action) { // Transformed-policy expected value minus original expected value.
      r += cfvs[phi_block_[action]] * policy[action]; // Accumulate the transformed action's CFV weighted by the original action's probability.
    }
    return r;
  }

 private:
  const std::vector<size_t> phi_block_; // Action mapping ai -> aj.

 public:
  const bool is_external_; // Whether the transformation is external.
};

using Sat = SwapActionTranformation; // Shorthand.

// WeightedActionTransformation handles weighted action transformations.
// Maintains the total weight weight_sum_ and a weighted phi matrix.
class WeightedActionTransformation {
 private:
 // Generate an (n+1)-by-n matrix with a negative identity in the first n rows and ones in the final row (index n).
  inline static Eigen::MatrixXd ProjectiveEye(size_t num_actions) {
    Eigen::MatrixXd a =
        -Eigen::MatrixXd::Identity(num_actions + 1, num_actions); // An (n+1)-by-n negative identity matrix, initially with zeros in the final row.
    a.row(num_actions) = Eigen::VectorXd::Constant(num_actions, 1.0); // Set every entry of the final row to 1.
    return a;
  }
  // Generate an (n+1)-dimensional constraint vector {0, 0, ..., 1}.
  inline static Eigen::VectorXd BVector(size_t num_actions) {
    Eigen::VectorXd b = Eigen::VectorXd::Zero(num_actions + 1); // Initialize a zero vector.
    b(num_actions) = 1.0; // Set the final entry to 1.
    return b;
  }

 public:
 // Construct from the number of actions and initialize the weighted matrix and vector.
  WeightedActionTransformation(size_t num_actions)
      : weight_sum_(0), // Initialize the total weight to 0.
        weighted_phi_blocks_(Eigen::MatrixXd::Zero(num_actions, num_actions)), // Initialize the weighted phi matrix as an n-by-n zero matrix.
        weighted_external_blocks_(Eigen::VectorXd::Zero(num_actions)), // Initialize the weighted external-transformation vector as an n-dimensional zero vector.
        all_external_(true),
        projective_eye_(ProjectiveEye(num_actions)), // Generate the projected identity matrix.
        b_(BVector(num_actions)) {} // Generate the constraint vector {0, 0, ..., 1}.
  virtual ~WeightedActionTransformation() = default;

  // Add a transformation phi with default weight 1.0.
  void Add(const Sat& phi, double weight = 1.0) {
    weight_sum_ += weight;
    if (phi.is_external_) {
      weighted_external_blocks_(phi.SwapTarget(0)) += weight; // All external-transformation entries share one target index; update only that action's weight.
    } else {
      all_external_ = false;
      for (size_t action = 0; action < NumActions(); ++action) {
        weighted_phi_blocks_(phi.SwapTarget(action), action) += weight; // For internal transformations, update the weight matrix at (aj, ai).
      }
    }
  }

  // Combine the internal-transformation weight matrix and external-transformation weight vector into the final transformation matrix.
  // e.g. [[m11+v1, m12+v1,...], [m21+v2, m22+v2,...],...]
  // weighted_external_blocks_ can be represented as a matrix with each row {vi, ...}, weighting mappings from every action to ai.
  Eigen::MatrixXd ToMatrix() const {
    return weighted_phi_blocks_.colwise() + weighted_external_blocks_; 
  }

  // Compute a fixed point of the transformation matrix and return it as a policy vector.
  std::vector<double> FixedPoint() const {
    std::vector<double> pi_mem(NumActions(), 1.0 / NumActions()); // Initialize an n-dimensional uniform distribution.
    FixedPoint(pi_mem);
    return pi_mem;
  }

  // Compute a fixed-point policy into pi_mem; each phi's weight is its weighted regret w^{t+1} * (rho(phi))^+.
  void FixedPoint(std::vector<double>& pi_mem) const {
    if (weight_sum_ > 0) {
      Eigen::Map<Eigen::VectorXd> pi(pi_mem.data(), NumActions()); // Map pi_mem into an Eigen vector pi; pi_mem.data() returns the underlying double*.
      if (all_external_) { // If all added transformations are external,
        pi = weighted_external_blocks_ / weight_sum_; // normalize the external-transformation vector directly.
      } else { // Otherwise, solve the linear system Ax=b.
        Eigen::MatrixXd A = projective_eye_; // Generate an (n+1)-by-n negative identity matrix.
        A.topRows(NumActions()) += ToMatrix() / weight_sum_; // Add the normalized transformation matrix to the columns.
        pi = (A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_)) // Solve the linear system Ax=b.
                 .cwiseMax(0)
                 .cwiseMin(1.0); // Keep the result in [0, 1]; clarify this normalization step.
      }
    } else {
      pi_mem.assign(NumActions(), 1.0 / NumActions()); // If the total weight is zero, return a uniform distribution.
    }
  }

  // Get the total weight.
  double WeightSum() const { return weight_sum_; }
  // Reset the weighted matrix and vector.
  void Reset() {
    weight_sum_ = 0;
    weighted_phi_blocks_.setZero();
    weighted_external_blocks_.setZero();
  }

  // Get the number of actions.
  size_t NumActions() const { return weighted_phi_blocks_.rows(); }

 private:
  double weight_sum_; // Total weight.
  Eigen::MatrixXd weighted_phi_blocks_; // Internal-transformation weight matrix.
  Eigen::VectorXd weighted_external_blocks_; // External-transformation weight vector.
  bool all_external_; // Whether all added transformations are external.
  const Eigen::MatrixXd projective_eye_; // An (n+1)-by-n negative projected identity matrix with ones in the final row.
  const Eigen::VectorXd b_; // An (n+1)-dimensional constraint vector {0, 0, ..., 1}.
};

}  // namespace hr_edl
#endif  // HR_EDL_ACTION_TRANSFORMATION_H_
