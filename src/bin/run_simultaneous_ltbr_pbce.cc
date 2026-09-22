#include <algorithm>
#include <cctype>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "bin/ltbr.h"
#include "hr_edl/adaptive_profile.h"
#include "hr_edl/decision_point.h"
#include "hr_edl/policy_evaluation.h"
#include "hr_edl/spiel_extra.h"
#include "hr_edl/stopwatch.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/spiel.h"

// Environment.
ABSL_FLAG(std::string, game, "leduc_poker", "The game to play.");
ABSL_FLAG(std::string, sampler, "null", "The sampler to use.");
ABSL_FLAG(int32_t, t, 500, "Maximum global iterations per algorithm.");
ABSL_FLAG(int32_t, random_seed, 0, "Seed for a sampler's random engine.");
ABSL_FLAG(bool, show_num, false, "Only show the number of algorithms.");
ABSL_FLAG(size_t, alg_group, 3, "Algorithm group (only PBCE group 3 is supported).");
ABSL_FLAG(double, epsilon, 0.0, "Fixed perturbation parameter.");
ABSL_FLAG(int32_t, threads, 1, "Number of algorithms to run concurrently.");
ABSL_FLAG(std::string, file_dir, "data/test_results.ssv", "Result file.");
ABSL_FLAG(bool, alt, true, "Use alternate/Gauss-Seidel updates.");
ABSL_FLAG(bool, cfr_plus, true, "Apply CFR+ truncation to cumulative solver regrets.");
ABSL_FLAG(bool, verbose_iteration_log, false, "Print every iteration to stdout.");
ABSL_FLAG(int64_t, iteration_log_interval, 0,
          "Print progress every N completed iterations; zero disables periodic logging.");

// Stagewise adaptive epsilon (PEFR).
ABSL_FLAG(bool, adaptive_epsilon, false, "Enable the stagewise PEFR controller.");
ABSL_FLAG(double, epsilon_init, 0.1, "Initial adaptive epsilon.");
ABSL_FLAG(double, epsilon_decay, 0.5, "Multiplicative epsilon decay.");
ABSL_FLAG(double, epsilon_min, 1e-5, "Smallest adaptive epsilon.");
ABSL_FLAG(double, stability_tolerance, 0.02,
          "Fixed relative-change and relative-variation tolerance for OSR/CSR stability.");
ABSL_FLAG(int32_t, stability_window, 5,
          "Checkpoints per rolling window used for plateau detection.");
ABSL_FLAG(double, min_response, 0.01,
          "Deprecated diagnostic: minimum relative OSR response after an epsilon update.");
ABSL_FLAG(double, response_ratio_tol, 0.05,
          "Deprecated diagnostic: CSR response as a fraction of the observed OSR response.");
ABSL_FLAG(double, quality_tolerance, 0.05,
          "Maximum relative OSR/CSR deterioration from the certified quality anchor.");
ABSL_FLAG(int64_t, checkpoint_freq, 500, "Adaptive metric/checkpoint frequency.");
ABSL_FLAG(int64_t, min_stage_iterations, 2000, "Minimum iterations in each stage.");
ABSL_FLAG(int32_t, stable_checkpoints, 3, "Consecutive ready checkpoints required.");
ABSL_FLAG(int64_t, max_stage_iterations, 0,
          "Maximum iterations in one stage; zero disables this limit.");
ABSL_FLAG(double, w_numerical_eps, 1e-12,
          "W values at or below this are treated as zero in CSR diagnostics.");
ABSL_FLAG(bool, save_stage_policies, false,
          "Serialize every current-stage joint behavioral profile to a sidecar JSONL file.");

namespace {
namespace fs = std::filesystem;

struct FixedRecord {
  double ev = 0.0;
  double milliseconds = 0.0;
  double posr_numerator = 0.0;
  double osr_numerator = 0.0;
  double csr = 0.0;
  double acsr = 0.0;
};

struct AdaptiveRecord {
  std::string record_type = "CHECKPOINT";
  std::string algorithm;
  int64_t global_t = 0;
  int stage_k = 0;
  int64_t stage_t = 0;
  double epsilon = 0.0;
  bool epsilon_changed = false;
  double posr = 0.0;
  double osr = 0.0;
  double csr = 0.0;
  double acsr = 0.0;
  double posr_norm = 0.0;
  double osr_norm = 0.0;
  double csr_norm = 0.0;
  double w_min = 0.0;
  double w_min_positive = 0.0;
  double w_mean = 0.0;
  double w_max = 0.0;
  double w_at_max_csr = 0.0;
  int64_t num_zero_or_tiny_w = 0;
  double osr_window_old = std::numeric_limits<double>::quiet_NaN();
  double osr_window_new = std::numeric_limits<double>::quiet_NaN();
  double csr_window_old = std::numeric_limits<double>::quiet_NaN();
  double csr_window_new = std::numeric_limits<double>::quiet_NaN();
  double osr_relative_change = std::numeric_limits<double>::quiet_NaN();
  double csr_relative_change = std::numeric_limits<double>::quiet_NaN();
  double osr_relative_variation = std::numeric_limits<double>::quiet_NaN();
  double csr_relative_variation = std::numeric_limits<double>::quiet_NaN();
  bool osr_stable = false;
  bool csr_stable = false;
  double response_osr_ref = std::numeric_limits<double>::quiet_NaN();
  double response_csr_ref = std::numeric_limits<double>::quiet_NaN();
  double osr_abs_response = 0.0;
  double osr_response = 0.0;
  double csr_response = 0.0;
  bool osr_direction_ok = false;
  bool csr_direction_ok = false;
  double max_osr_response = 0.0;
  double max_csr_response = 0.0;
  double required_csr_response = 0.0;
  double response_ratio = 0.0;
  bool osr_response_ready = false;
  bool csr_response_ready = false;
  bool response_ready = false;
  bool enough_history = false;
  bool enough_stage_iterations = false;
  double quality_anchor_osr = std::numeric_limits<double>::quiet_NaN();
  double quality_anchor_csr = std::numeric_limits<double>::quiet_NaN();
  double current_quality_osr = std::numeric_limits<double>::quiet_NaN();
  double current_quality_csr = std::numeric_limits<double>::quiet_NaN();
  double osr_quality_ratio = std::numeric_limits<double>::quiet_NaN();
  double csr_quality_ratio = std::numeric_limits<double>::quiet_NaN();
  bool osr_quality_safe = false;
  bool csr_quality_safe = false;
  bool quality_safe = false;
  bool quality_anchor_updated = false;
  bool stage_ready = false;
  int stable_count = 0;
  bool stage_certified = false;
  double milliseconds = 0.0;
  double old_epsilon = 0.0;
  double new_epsilon = 0.0;
  std::string status;
};

std::string CsvEscape(const std::string& value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
  std::string escaped = "\"";
  for (char c : value) {
    if (c == '\"') escaped += '\"';
    escaped += c;
  }
  escaped += "\"";
  return escaped;
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
      case '\\':
        out << "\\\\";
        break;
      case '\"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c)
              << std::dec;
        } else {
          out << c;
        }
    }
  }
  return out.str();
}

std::string SafeLabel(std::string label) {
  for (char& c : label) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') c = '_';
  }
  return label;
}

class StagePolicyWriter {
 public:
  StagePolicyWriter(const fs::path& result_path, const std::string& algorithm, bool enabled)
      : directory_(result_path.string() + ".policies"), algorithm_(algorithm), enabled_(enabled) {
    if (enabled_) fs::create_directories(directory_);
  }

  void Start(int stage_k, double epsilon) {
    if (!enabled_) return;
    if (out_.is_open()) throw std::logic_error("stage policy writer already open");
    std::ostringstream epsilon_text;
    epsilon_text << std::setprecision(17) << epsilon;
    final_path_ = directory_ / (SafeLabel(algorithm_) + ".stage_" + std::to_string(stage_k) +
                                ".epsilon_" + SafeLabel(epsilon_text.str()) + ".jsonl");
    temp_path_ = final_path_;
    temp_path_ += ".tmp";
    out_.open(temp_path_, std::ios::out | std::ios::trunc);
    if (!out_) throw std::runtime_error("cannot open stage policy file: " + temp_path_.string());
    profile_count_ = 0;
    out_ << "{\"type\":\"metadata\",\"algorithm\":\"" << JsonEscape(algorithm_)
         << "\",\"stage_k\":" << stage_k << ",\"epsilon\":" << std::setprecision(17) << epsilon
         << ",\"semantics\":\"uniform time-selection over the PROFILE records in this file\"}\n";
  }

  void Write(int64_t global_t, int64_t stage_t, hr_edl::DecisionPoint& root,
             const hr_edl::PolicyProfile& profile) {
    if (!enabled_) return;
    using Key = std::pair<int, std::string>;
    std::map<Key, open_spiel::ActionsAndProbs> policies;
    hr_edl::ForEachState(
        root,
        [&policies, &profile](const hr_edl::DecisionPoint& dp) {
          const int player = dp.PlayerToAct();
          if (player < 0) return;
          Key key{player, dp.InformationStateString()};
          if (policies.find(key) == policies.end()) {
            policies.emplace(std::move(key),
                             profile[player]->GetStatePolicy(*dp.OpenSpielStatePtr()));
          }
        },
        hr_edl::ALL_PLAYERS);

    out_ << "{\"type\":\"PROFILE\",\"global_t\":" << global_t << ",\"stage_t\":" << stage_t
         << ",\"policies\":[";
    bool first_policy = true;
    for (const auto& entry : policies) {
      if (!first_policy) out_ << ',';
      first_policy = false;
      out_ << "{\"player\":" << entry.first.first << ",\"info_state\":\""
           << JsonEscape(entry.first.second) << "\",\"actions\":[";
      bool first_action = true;
      for (const auto& action_and_prob : entry.second) {
        if (!first_action) out_ << ',';
        first_action = false;
        out_ << '[' << action_and_prob.first << ',' << std::setprecision(17)
             << action_and_prob.second << ']';
      }
      out_ << "]}";
    }
    out_ << "]}\n";
    if (!out_) throw std::runtime_error("failed writing stage policy history");
    ++profile_count_;
  }

  void Finish(bool certified, int64_t expected_profiles) {
    if (!enabled_ || !out_.is_open()) return;
    if (profile_count_ != expected_profiles) {
      throw std::runtime_error("stage policy history count does not match stage_t");
    }
    out_.flush();
    if (!out_) throw std::runtime_error("failed flushing stage policy history");
    out_.close();
    fs::path destination = final_path_;
    if (!certified) destination += ".partial";
    std::error_code error;
    fs::rename(temp_path_, destination, error);
    if (error) {
      throw std::runtime_error("failed to finalize stage policy history: " + error.message());
    }
  }

  ~StagePolicyWriter() {
    if (out_.is_open()) out_.close();
  }

 private:
  fs::path directory_;
  std::string algorithm_;
  bool enabled_;
  fs::path temp_path_;
  fs::path final_path_;
  std::ofstream out_;
  int64_t profile_count_ = 0;
};

size_t MaxInformationSetActions(hr_edl::DecisionPoint& root) {
  size_t max_actions = 0;
  hr_edl::ForEachState(
      root,
      [&max_actions](const hr_edl::DecisionPoint& dp) {
        if (dp.PlayerToAct() >= 0) max_actions = std::max(max_actions, dp.NumActions());
      },
      hr_edl::ALL_PLAYERS);
  return max_actions;
}

struct AdaptiveControllerState {
  std::deque<double> osr_history;
  std::deque<double> csr_history;
  double response_osr_ref = std::numeric_limits<double>::quiet_NaN();
  double response_csr_ref = std::numeric_limits<double>::quiet_NaN();
  bool response_ref_initialized = false;
  double max_osr_response = 0.0;
  double max_csr_response = 0.0;
  double quality_anchor_osr = std::numeric_limits<double>::quiet_NaN();
  double quality_anchor_csr = std::numeric_limits<double>::quiet_NaN();
  bool quality_anchor_initialized = false;
  int stable_count = 0;

  void SetResponseReference(double osr, double csr) {
    response_osr_ref = osr;
    response_csr_ref = csr;
    response_ref_initialized = true;
  }

  void ResetStage() {
    osr_history.clear();
    csr_history.clear();
    max_osr_response = 0.0;
    max_csr_response = 0.0;
    stable_count = 0;
  }

  bool MaybeUpdateQualityAnchor(double osr, double csr) {
    if (!quality_anchor_initialized ||
        (osr <= quality_anchor_osr && csr <= quality_anchor_csr)) {
      quality_anchor_osr = osr;
      quality_anchor_csr = csr;
      quality_anchor_initialized = true;
      return true;
    }
    return false;
  }
};

double Median(std::vector<double> values) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  if (values.size() % 2 != 0) return values[middle];
  return 0.5 * (values[middle - 1] + values[middle]);
}

std::vector<double> Window(const std::deque<double>& history, size_t begin, size_t end) {
  return std::vector<double>(history.begin() + begin, history.begin() + end);
}

double RelativeMad(const std::vector<double>& values, double median, double numerical_eps) {
  std::vector<double> deviations;
  deviations.reserve(values.size());
  for (double value : values) deviations.push_back(std::abs(value - median));
  return Median(std::move(deviations)) / std::max(std::abs(median), numerical_eps);
}

bool IsNonnegativeFinite(double value) { return std::isfinite(value) && value >= 0.0; }

bool ValidateConfiguration(size_t max_actions) {
  const bool adaptive = absl::GetFlag(FLAGS_adaptive_epsilon);
  const double epsilon =
      adaptive ? absl::GetFlag(FLAGS_epsilon_init) : absl::GetFlag(FLAGS_epsilon);
  if (max_actions == 0) {
    std::cerr << "No player information set with a legal action was found.\n";
    return false;
  }
  if (!std::isfinite(epsilon) || epsilon < 0.0 ||
      epsilon * static_cast<double>(max_actions) >= 1.0) {
    std::cerr << "Invalid epsilon=" << epsilon << ": this game has an information set with "
              << max_actions << " actions, so epsilon must satisfy 0 <= epsilon < "
              << 1.0 / max_actions << ".\n";
    return false;
  }
  if (absl::GetFlag(FLAGS_t) <= 0 || absl::GetFlag(FLAGS_threads) <= 0) {
    std::cerr << "--t and --threads must be positive.\n";
    return false;
  }
  if (absl::GetFlag(FLAGS_iteration_log_interval) < 0) {
    std::cerr << "--iteration_log_interval must be non-negative.\n";
    return false;
  }
  if (!adaptive) return true;
  if (absl::GetFlag(FLAGS_alg_group) != 3) {
    std::cerr << "--adaptive_epsilon currently requires --alg_group=3.\n";
    return false;
  }
  const double epsilon_min = absl::GetFlag(FLAGS_epsilon_min);
  const double epsilon_decay = absl::GetFlag(FLAGS_epsilon_decay);
  if (!std::isfinite(epsilon_min) || epsilon_min < 0.0 || epsilon_min > epsilon) {
    std::cerr << "--epsilon_min must be finite, non-negative, and <= --epsilon_init.\n";
    return false;
  }
  if (!(epsilon_decay > 0.0 && epsilon_decay < 1.0)) {
    std::cerr << "--epsilon_decay must be in (0,1).\n";
    return false;
  }
  if (absl::GetFlag(FLAGS_checkpoint_freq) <= 0 || absl::GetFlag(FLAGS_min_stage_iterations) < 0 ||
      absl::GetFlag(FLAGS_stable_checkpoints) <= 0 || absl::GetFlag(FLAGS_stability_window) < 2 ||
      absl::GetFlag(FLAGS_max_stage_iterations) < 0 ||
      !IsNonnegativeFinite(absl::GetFlag(FLAGS_w_numerical_eps)) ||
      !IsNonnegativeFinite(absl::GetFlag(FLAGS_stability_tolerance)) ||
      !IsNonnegativeFinite(absl::GetFlag(FLAGS_quality_tolerance)) ||
      !IsNonnegativeFinite(absl::GetFlag(FLAGS_min_response)) ||
      !IsNonnegativeFinite(absl::GetFlag(FLAGS_response_ratio_tol))) {
    std::cerr << "Adaptive counts and thresholds are invalid; stability_window must be at "
                 "least two and checkpoint/patience counts must be positive.\n";
    return false;
  }
  if (absl::GetFlag(FLAGS_response_ratio_tol) > 1.0) {
    std::cerr << "--response_ratio_tol must lie in [0,1].\n";
    return false;
  }
  return true;
}

void AppendBounded(std::deque<double>* history, double value, size_t max_size) {
  history->push_back(value);
  if (history->size() > max_size) history->pop_front();
}

AdaptiveRecord MakeMetricRecord(const std::string& algorithm, int64_t global_t, int stage_k,
                                int64_t stage_t, double epsilon, double utility_scale,
                                double milliseconds, const hr_edl::StageRegretMetrics& metrics,
                                int stable_count) {
  AdaptiveRecord record;
  record.algorithm = algorithm;
  record.global_t = global_t;
  record.stage_k = stage_k;
  record.stage_t = stage_t;
  record.epsilon = epsilon;
  record.posr = metrics.max_transformed_full_regret / stage_t;
  record.osr = metrics.max_original_full_regret / stage_t;
  record.csr = metrics.max_conditional_regret;
  record.acsr = metrics.sum_positive_conditional_regret;
  record.posr_norm = record.posr / utility_scale;
  record.osr_norm = record.osr / utility_scale;
  record.csr_norm = record.csr / utility_scale;
  record.w_min = metrics.w_min;
  record.w_min_positive = metrics.w_min_positive;
  record.w_mean = metrics.WMean();
  record.w_max = metrics.w_max;
  record.w_at_max_csr = metrics.w_at_max_csr;
  record.num_zero_or_tiny_w = metrics.num_zero_or_tiny_w;
  record.stable_count = stable_count;
  record.milliseconds = milliseconds;
  return record;
}

bool IsOSRResponseReady(int stage_k, const AdaptiveControllerState& state) {
  if (stage_k == 0) return true;
  return state.response_ref_initialized &&
         state.max_osr_response >= absl::GetFlag(FLAGS_min_response);
}

bool IsCSRResponseReady(int stage_k, const AdaptiveControllerState& state,
                        double csr_response) {
  if (stage_k == 0) return true;
  return state.response_ref_initialized &&
         csr_response >=
             absl::GetFlag(FLAGS_response_ratio_tol) * state.max_osr_response;
}

void UpdateStageResponse(AdaptiveControllerState* state, AdaptiveRecord* record) {
  const double numerical_eps = std::numeric_limits<double>::epsilon();
  if (state->response_ref_initialized) {
    const double osr_denom =
        std::max(std::abs(state->response_osr_ref), numerical_eps);
    record->osr_abs_response =
        std::abs(record->osr - state->response_osr_ref) / osr_denom;
    record->osr_response =
        std::max((state->response_osr_ref - record->osr) / osr_denom, 0.0);
    record->csr_response =
        (state->response_csr_ref - record->csr) /
        std::max(std::abs(state->response_csr_ref), numerical_eps);
    record->osr_direction_ok = record->osr <= state->response_osr_ref;
    record->csr_direction_ok = record->csr <= state->response_csr_ref;
    state->max_osr_response = std::max(state->max_osr_response, record->osr_response);
    state->max_csr_response = std::max(state->max_csr_response, record->csr_response);
  }

  record->response_osr_ref = state->response_osr_ref;
  record->response_csr_ref = state->response_csr_ref;
  record->max_osr_response = state->max_osr_response;
  record->max_csr_response = state->max_csr_response;
  record->required_csr_response =
      absl::GetFlag(FLAGS_response_ratio_tol) * state->max_osr_response;
  // Diagnostic only; CSR readiness compares the current CSR response with
  // required_csr_response below.
  record->response_ratio =
      state->max_osr_response > numerical_eps
          ? state->max_csr_response / state->max_osr_response
          : 0.0;
  record->osr_response_ready = IsOSRResponseReady(record->stage_k, *state);
  record->csr_response_ready =
      IsCSRResponseReady(record->stage_k, *state, record->csr_response);
  record->response_ready = record->osr_response_ready && record->csr_response_ready;
}

void UpdateQualityDiagnostics(const AdaptiveControllerState& state, AdaptiveRecord* record) {
  record->quality_anchor_osr = state.quality_anchor_osr;
  record->quality_anchor_csr = state.quality_anchor_csr;
  if (!record->enough_history) {
    record->quality_safe = !state.quality_anchor_initialized;
    return;
  }

  record->current_quality_osr = record->osr_window_new;
  record->current_quality_csr = record->csr_window_new;
  if (!state.quality_anchor_initialized) {
    record->osr_quality_safe = true;
    record->csr_quality_safe = true;
    record->quality_safe = true;
    return;
  }

  const double numerical_eps = std::numeric_limits<double>::epsilon();
  const double tolerance = absl::GetFlag(FLAGS_quality_tolerance);
  const double osr_quality_limit =
      state.quality_anchor_osr +
      tolerance * std::max(std::abs(state.quality_anchor_osr), numerical_eps);
  const double csr_quality_limit =
      state.quality_anchor_csr +
      tolerance * std::max(std::abs(state.quality_anchor_csr), numerical_eps);
  record->osr_quality_ratio =
      record->current_quality_osr /
      std::max(std::abs(state.quality_anchor_osr), numerical_eps);
  record->csr_quality_ratio =
      record->current_quality_csr /
      std::max(std::abs(state.quality_anchor_csr), numerical_eps);
  record->osr_quality_safe = record->current_quality_osr <= osr_quality_limit;
  record->csr_quality_safe = record->current_quality_csr <= csr_quality_limit;
  record->quality_safe = record->osr_quality_safe && record->csr_quality_safe;
}

bool IsOSRStable(const AdaptiveRecord& record) {
  const double tolerance = absl::GetFlag(FLAGS_stability_tolerance);
  return record.osr_relative_change <= tolerance && record.osr_relative_variation <= tolerance;
}

bool IsCSRStable(const AdaptiveRecord& record) {
  const double tolerance = absl::GetFlag(FLAGS_stability_tolerance);
  return record.csr_relative_change <= tolerance && record.csr_relative_variation <= tolerance;
}

void UpdateRollingDiagnostics(AdaptiveControllerState* state, AdaptiveRecord* record) {
  const size_t window_size = static_cast<size_t>(absl::GetFlag(FLAGS_stability_window));
  const size_t history_limit = 2 * window_size;
  AppendBounded(&state->osr_history, record->osr_norm, history_limit);
  AppendBounded(&state->csr_history, record->csr_norm, history_limit);
  if (state->osr_history.size() < history_limit) return;

  const auto old_osr = Window(state->osr_history, 0, window_size);
  const auto new_osr = Window(state->osr_history, window_size, history_limit);
  const auto old_csr = Window(state->csr_history, 0, window_size);
  const auto new_csr = Window(state->csr_history, window_size, history_limit);
  record->osr_window_old = Median(old_osr);
  record->osr_window_new = Median(new_osr);
  record->csr_window_old = Median(old_csr);
  record->csr_window_new = Median(new_csr);

  const double numerical_eps = std::numeric_limits<double>::epsilon();
  record->osr_relative_change = std::abs(record->osr_window_new - record->osr_window_old) /
                                std::max(std::abs(record->osr_window_old), numerical_eps);
  record->csr_relative_change = std::abs(record->csr_window_new - record->csr_window_old) /
                                std::max(std::abs(record->csr_window_old), numerical_eps);
  record->osr_relative_variation = RelativeMad(new_osr, record->osr_window_new, numerical_eps);
  record->csr_relative_variation = RelativeMad(new_csr, record->csr_window_new, numerical_eps);

  record->osr_stable = IsOSRStable(*record);
  record->csr_stable = IsCSRStable(*record);
  record->enough_history = true;
}

void SetStageReadiness(AdaptiveRecord* record) {
  record->enough_stage_iterations =
      record->stage_t >= absl::GetFlag(FLAGS_min_stage_iterations);
  // Response diagnostics are retained for CLI/output compatibility, but no
  // longer gate epsilon-stage certification.
  record->stage_ready = record->enough_history && record->enough_stage_iterations &&
                        record->osr_stable && record->csr_stable && record->quality_safe;
}

bool NumericallyValid(const AdaptiveRecord& r) {
  return std::isfinite(r.epsilon) && std::isfinite(r.posr) && std::isfinite(r.osr) &&
         std::isfinite(r.csr) && std::isfinite(r.acsr) && std::isfinite(r.posr_norm) &&
         std::isfinite(r.osr_norm) && std::isfinite(r.csr_norm) && std::isfinite(r.w_min) &&
         std::isfinite(r.w_min_positive) && std::isfinite(r.w_mean) && std::isfinite(r.w_max) &&
         std::isfinite(r.w_at_max_csr);
}

void PrintCheckpoint(const AdaptiveRecord& r) {
  std::cout << "CHECKPOINT algorithm=" << r.algorithm << " global_t=" << r.global_t
            << " stage=" << r.stage_k << " stage_t=" << r.stage_t << " epsilon=" << r.epsilon
            << " response_osr_ref=" << r.response_osr_ref
            << " response_csr_ref=" << r.response_csr_ref << " osr=" << r.osr
            << " csr=" << r.csr << " osr_abs_response=" << r.osr_abs_response
            << " osr_response=" << r.osr_response
            << " csr_response=" << r.csr_response
            << " osr_direction_ok=" << r.osr_direction_ok
            << " csr_direction_ok=" << r.csr_direction_ok
            << " osr_stable=" << r.osr_stable
            << " csr_stable=" << r.csr_stable
            << " max_osr_response=" << r.max_osr_response
            << " max_csr_response=" << r.max_csr_response
            << " required_csr_response=" << r.required_csr_response
            << " response_ratio=" << r.response_ratio
            << " osr_response_ready=" << r.osr_response_ready
            << " csr_response_ready=" << r.csr_response_ready
            << " response_ready=" << r.response_ready
            << " enough_history=" << r.enough_history
            << " enough_stage_iterations=" << r.enough_stage_iterations
            << " quality_anchor_osr=" << r.quality_anchor_osr
            << " quality_anchor_csr=" << r.quality_anchor_csr
            << " current_quality_osr=" << r.current_quality_osr
            << " current_quality_csr=" << r.current_quality_csr
            << " osr_quality_ratio=" << r.osr_quality_ratio
            << " csr_quality_ratio=" << r.csr_quality_ratio
            << " osr_quality_safe=" << r.osr_quality_safe
            << " csr_quality_safe=" << r.csr_quality_safe
            << " quality_safe=" << r.quality_safe
            << " quality_anchor_updated=" << r.quality_anchor_updated
            << " stage_ready=" << r.stage_ready << " stable_count=" << r.stable_count
            << " certified=" << r.stage_certified << '\n';
}

void WriteAdaptiveCsv(std::ostream& out,
                      const std::vector<std::vector<AdaptiveRecord>>& all_records) {
  out << "record_type,algorithm,global_t,epsilon_stage,stage_k,stage_t,stage_round,epsilon,"
         "epsilon_changed,posr,osr,csr,acsr,posr_norm,osr_norm,csr_norm,w_min,"
         "w_min_positive,w_mean,w_max,w_at_max_csr,num_zero_or_tiny_W,osr_window_old,"
         "osr_window_new,csr_window_old,csr_window_new,osr_relative_change,"
         "csr_relative_change,osr_relative_variation,csr_relative_variation,osr_stable,"
         "csr_stable,response_osr_ref,response_csr_ref,osr_abs_response,osr_response,"
         "csr_response,"
         "osr_direction_ok,csr_direction_ok,max_osr_response,max_csr_response,"
         "required_csr_response,response_ratio,osr_response_ready,csr_response_ready,"
         "response_ready,enough_history,enough_stage_iterations,quality_anchor_osr,"
         "quality_anchor_csr,current_quality_osr,current_quality_csr,osr_quality_ratio,"
         "csr_quality_ratio,osr_quality_safe,csr_quality_safe,quality_safe,"
         "quality_anchor_updated,"
         "stage_ready,stable_count,"
         "stage_certified,milliseconds,old_epsilon,new_epsilon,status\n";
  out << std::setprecision(17);
  for (const auto& records : all_records) {
    for (const auto& r : records) {
      out << r.record_type << ',' << CsvEscape(r.algorithm) << ',' << r.global_t << ',' << r.stage_k
          << ',' << r.stage_k << ',' << r.stage_t << ',' << r.stage_t << ',' << r.epsilon << ','
          << r.epsilon_changed << ',' << r.posr << ',' << r.osr << ',' << r.csr << ',' << r.acsr
          << ',' << r.posr_norm << ',' << r.osr_norm << ',' << r.csr_norm << ',' << r.w_min << ','
          << r.w_min_positive << ',' << r.w_mean << ',' << r.w_max << ',' << r.w_at_max_csr << ','
          << r.num_zero_or_tiny_w << ',' << r.osr_window_old << ',' << r.osr_window_new << ','
          << r.csr_window_old << ',' << r.csr_window_new << ',' << r.osr_relative_change << ','
          << r.csr_relative_change << ',' << r.osr_relative_variation << ','
          << r.csr_relative_variation << ',' << r.osr_stable << ',' << r.csr_stable << ','
          << r.response_osr_ref << ',' << r.response_csr_ref << ',' << r.osr_abs_response << ','
          << r.osr_response << ',' << r.csr_response << ',' << r.osr_direction_ok << ','
          << r.csr_direction_ok << ',' << r.max_osr_response << ',' << r.max_csr_response << ','
          << r.required_csr_response << ',' << r.response_ratio << ','
          << r.osr_response_ready << ','
          << r.csr_response_ready << ',' << r.response_ready << ','
          << r.enough_history << ',' << r.enough_stage_iterations << ','
          << r.quality_anchor_osr << ',' << r.quality_anchor_csr << ','
          << r.current_quality_osr << ',' << r.current_quality_csr << ','
          << r.osr_quality_ratio << ',' << r.csr_quality_ratio << ','
          << r.osr_quality_safe << ',' << r.csr_quality_safe << ',' << r.quality_safe << ','
          << r.quality_anchor_updated << ',' << r.stage_ready << ',' << r.stable_count << ','
          << r.stage_certified << ',' << r.milliseconds << ',' << r.old_epsilon << ','
          << r.new_epsilon << ',' << r.status << '\n';
    }
  }
}

}  // namespace

bool run_experiment() {
  const std::string game_name = absl::GetFlag(FLAGS_game);
  std::shared_ptr<const open_spiel::Game> game;
  if (absl::StrContains(game_name, ".efg")) {
    if (!fs::exists(game_name)) {
      std::cerr << "Error: File does not exist: " << game_name << '\n';
      return false;
    }
    game = open_spiel::LoadGameAsTurnBased("efg_game",
                                           {{"filename", open_spiel::GameParameter(game_name)}});
  } else {
    game = open_spiel::LoadGameAsTurnBased(game_name);
  }
  if (!game) {
    std::cerr << "Error: Failed to load game: " << game_name << '\n';
    return false;
  }

  if (absl::GetFlag(FLAGS_alg_group) != 3) {
    std::cerr << "This executable supports only PBCE (--alg_group=3).\n";
    return false;
  }
  const bool adaptive = absl::GetFlag(FLAGS_adaptive_epsilon);
  const double initial_epsilon =
      adaptive ? absl::GetFlag(FLAGS_epsilon_init) : absl::GetFlag(FLAGS_epsilon);
  const auto labeled_algs = ltbr_perturbed_cfr_algs(initial_epsilon);
  const auto alg_labels = LearnerProfileLabels(labeled_algs);
  if (absl::GetFlag(FLAGS_show_num)) {
    std::cout << alg_labels.size() << '\n';
    return true;
  }

  hr_edl::CachedDecisionPoint root(game->NewInitialState(), false, true);
  const size_t max_actions = MaxInformationSetActions(root);
  if (!ValidateConfiguration(max_actions)) return false;

  const int64_t iterations = absl::GetFlag(FLAGS_t);
  const int num_threads = absl::GetFlag(FLAGS_threads);
  const int random_seed = absl::GetFlag(FLAGS_random_seed);
  const std::string sampler_name = absl::GetFlag(FLAGS_sampler);
  const fs::path output_path(absl::GetFlag(FLAGS_file_dir));
  const bool alt = absl::GetFlag(FLAGS_alt);
  const bool cfr_plus = absl::GetFlag(FLAGS_cfr_plus);
  const bool verbose = absl::GetFlag(FLAGS_verbose_iteration_log);
  const int64_t iteration_log_interval = absl::GetFlag(FLAGS_iteration_log_interval);
  const double utility_diameter = game->MaxUtility() - game->MinUtility();
  const double utility_scale =
      utility_diameter > std::numeric_limits<double>::epsilon() ? utility_diameter : 1.0;

  std::vector<std::vector<FixedRecord>> fixed_records(
      alg_labels.size(), std::vector<FixedRecord>(adaptive ? 0 : iterations));
  std::vector<std::vector<AdaptiveRecord>> adaptive_records(alg_labels.size());
  // Do not use vector<bool> here: algorithm threads write distinct entries and
  // bit-packing would still create a shared-word data race.
  std::vector<int> run_end_seen(alg_labels.size(), 0);
  std::vector<int> process_ok(alg_labels.size(), 1);
  std::queue<std::thread> threads;
  std::mutex iteration_log_mutex;

  for (size_t alg = 0; alg < alg_labels.size(); ++alg) {
    const auto task = [&, alg]() {
      const std::string& label = alg_labels[alg];
      try {
        hr_edl::CachedDecisionPoint local_root(root);
        auto sampler = hr_edl::NewSampler(sampler_name, random_seed);
        auto learner = labeled_algs[alg].New(local_root.NumPlayers(), utility_diameter);
        learner->SetCfrPlus(cfr_plus);
        hr_edl::Stopwatch stopwatch;

        if (!adaptive) {
          for (int64_t t = 0; t < iterations; ++t) {
            if (verbose) {
              std::lock_guard<std::mutex> lock(iteration_log_mutex);
              std::cout << "Iteration " << t << " algorithm=" << label << '\n';
            }
            const auto frozen = learner->Frozen();
            stopwatch.reset();
            const double ev = t < iterations - 1
                                  ? (alt ? learner->UpdateAlternateAndReturnEv(local_root, *sampler)
                                         : learner->UpdateAndReturnEv(local_root, *sampler, frozen))
                                  : learner->Ev(local_root, *sampler, frozen);
            const double elapsed = stopwatch.milliseconds();
            const auto regrets = learner->PhiRegrets();
            fixed_records[alg][t] = {ev,
                                     elapsed,
                                     std::get<0>(regrets),
                                     std::get<1>(regrets),
                                     std::get<2>(regrets),
                                     std::get<3>(regrets)};
            if (!verbose && iteration_log_interval > 0 &&
                ((t + 1) % iteration_log_interval == 0 || t + 1 == iterations)) {
              std::lock_guard<std::mutex> lock(iteration_log_mutex);
              std::cout << "Progress iteration=" << t + 1 << '/' << iterations
                        << " algorithm=" << label << std::endl;
            }
          }
          run_end_seen[alg] = true;
          return;
        }

        int stage_k = 0;
        int64_t stage_t = 0;
        int64_t global_t = 0;
        AdaptiveControllerState controller;
        double epsilon = initial_epsilon;
        double cumulative_milliseconds = 0.0;
        bool finished = false;
        std::string final_status;
        AdaptiveRecord last_record;
        last_record.algorithm = label;
        last_record.epsilon = epsilon;
        learner->ResetStageEvaluation();
        StagePolicyWriter policy_writer(output_path, label,
                                        absl::GetFlag(FLAGS_save_stage_policies));
        policy_writer.Start(stage_k, epsilon);

        while (!finished && global_t < iterations) {
          ++global_t;
          ++stage_t;
          if (verbose) {
            std::lock_guard<std::mutex> lock(iteration_log_mutex);
            std::cout << "Iteration algorithm=" << label << " global_t=" << global_t
                      << " stage=" << stage_k << " stage_t=" << stage_t << " epsilon=" << epsilon
                      << '\n';
          }
          const auto frozen_before_update = learner->Frozen();
          stopwatch.reset();
          if (alt) {
            learner->UpdateAlternateAndReturnEv(local_root, *sampler);
          } else {
            learner->UpdateAndReturnEv(local_root, *sampler, frozen_before_update);
          }
          cumulative_milliseconds += stopwatch.milliseconds();
          if (!verbose && iteration_log_interval > 0 &&
              (global_t % iteration_log_interval == 0 || global_t == iterations)) {
            std::lock_guard<std::mutex> lock(iteration_log_mutex);
            std::cout << "Progress global_t=" << global_t << '/' << iterations
                      << " algorithm=" << label << " stage=" << stage_k << " stage_t=" << stage_t
                      << " epsilon=" << epsilon << std::endl;
          }

          const auto current_profile = learner->Frozen();
          policy_writer.Write(global_t, stage_t, local_root, current_profile);

          const bool regular_checkpoint = stage_t % absl::GetFlag(FLAGS_checkpoint_freq) == 0;
          const bool stage_timeout = absl::GetFlag(FLAGS_max_stage_iterations) > 0 &&
                                     stage_t >= absl::GetFlag(FLAGS_max_stage_iterations);
          const bool global_limit = global_t >= iterations;
          if (!regular_checkpoint && !stage_timeout && !global_limit) continue;

          auto metrics = learner->StageMetrics(absl::GetFlag(FLAGS_w_numerical_eps));
          auto record = MakeMetricRecord(label, global_t, stage_k, stage_t, epsilon, utility_scale,
                                         cumulative_milliseconds, metrics, controller.stable_count);
          if (!NumericallyValid(record)) {
            process_ok[alg] = false;
            record.status = "NUMERICAL_ERROR";
            adaptive_records[alg].push_back(record);
            last_record = record;
            PrintCheckpoint(record);
            final_status = "NUMERICAL_ERROR";
            policy_writer.Finish(false, stage_t);
            finished = true;
            continue;
          }
          const bool at_epsilon_min =
              epsilon <= absl::GetFlag(FLAGS_epsilon_min) + std::numeric_limits<double>::epsilon() *
                                                                std::max(1.0, std::abs(epsilon));
          UpdateStageResponse(&controller, &record);
          if (regular_checkpoint) {
            UpdateRollingDiagnostics(&controller, &record);
          }
          UpdateQualityDiagnostics(controller, &record);
          SetStageReadiness(&record);
          if (regular_checkpoint) {
            controller.stable_count = record.stage_ready ? controller.stable_count + 1 : 0;
          }
          record.stable_count = controller.stable_count;
          const bool certified = regular_checkpoint &&
                                 controller.stable_count >= absl::GetFlag(FLAGS_stable_checkpoints);
          record.stage_certified = certified;
          record.epsilon_changed = certified && !at_epsilon_min;
          if (certified) {
            record.quality_anchor_updated = controller.MaybeUpdateQualityAnchor(
                record.current_quality_osr, record.current_quality_csr);
            // Keep the certified row and switch log consistent with the
            // cross-stage anchor that will be carried into the next stage.
            UpdateQualityDiagnostics(controller, &record);
          }
          if (!certified && (stage_timeout || global_limit)) {
            record.status = at_epsilon_min ? "FINAL_NOT_CERTIFIED"
                                           : (stage_timeout ? "STAGE_TIMEOUT" : "MAX_GLOBAL_ITER");
          }
          adaptive_records[alg].push_back(record);
          last_record = record;
          PrintCheckpoint(record);

          if (certified) {
            AdaptiveRecord stage_end = record;
            stage_end.record_type = "STAGE_END";
            stage_end.status = at_epsilon_min ? "CERTIFIED_FINAL" : "CERTIFIED_STAGE";
            adaptive_records[alg].push_back(stage_end);
            std::cout << "STAGE_END algorithm=" << label << " stage=" << stage_k
                      << " epsilon=" << epsilon << " stage_t=" << stage_t
                      << " global_t=" << global_t
                      << " final_osr=" << record.osr << " final_csr=" << record.csr
                      << " osr_abs_response=" << record.osr_abs_response
                      << " osr_response=" << record.osr_response
                      << " csr_response=" << record.csr_response
                      << " osr_direction_ok=" << record.osr_direction_ok
                      << " csr_direction_ok=" << record.csr_direction_ok
                      << " osr_response_ready=" << record.osr_response_ready
                      << " csr_response_ready=" << record.csr_response_ready
                      << " max_osr_response=" << record.max_osr_response
                      << " max_csr_response=" << record.max_csr_response
                      << " required_csr_response=" << record.required_csr_response
                      << " response_ratio=" << record.response_ratio
                      << " response_ready=" << record.response_ready
                      << " certified_osr=" << record.current_quality_osr
                      << " certified_csr=" << record.current_quality_csr
                      << " quality_anchor_osr=" << record.quality_anchor_osr
                      << " quality_anchor_csr=" << record.quality_anchor_csr
                      << " quality_safe=" << record.quality_safe
                      << " quality_anchor_updated=" << record.quality_anchor_updated << '\n';
            policy_writer.Finish(true, stage_t);
            if (at_epsilon_min) {
              final_status = "CERTIFIED_FINAL";
              finished = true;
              continue;
            }

            const double old_epsilon = epsilon;
            controller.SetResponseReference(record.osr, record.csr);
            epsilon = std::max(absl::GetFlag(FLAGS_epsilon_min),
                               absl::GetFlag(FLAGS_epsilon_decay) * epsilon);
            AdaptiveRecord update = stage_end;
            update.record_type = "EPSILON_UPDATE";
            update.old_epsilon = old_epsilon;
            update.new_epsilon = epsilon;
            update.epsilon_changed = true;
            update.response_osr_ref = controller.response_osr_ref;
            update.response_csr_ref = controller.response_csr_ref;
            adaptive_records[alg].push_back(update);
            std::cout << "EPSILON_UPDATE algorithm=" << label << " stage=" << stage_k
                      << " old_epsilon=" << old_epsilon << " new_epsilon=" << epsilon
                      << " global_t=" << global_t
                      << " response_osr_ref=" << controller.response_osr_ref
                      << " response_csr_ref=" << controller.response_csr_ref
                      << " certified_osr=" << record.current_quality_osr
                      << " certified_csr=" << record.current_quality_csr
                      << " quality_anchor_osr=" << controller.quality_anchor_osr
                      << " quality_anchor_csr=" << controller.quality_anchor_csr
                      << " quality_anchor_updated=" << record.quality_anchor_updated << '\n';
            const double solver_regret_before = learner->SolverRegretL1();
            const double stage_original_before = learner->StageOriginalRegretL1();
            const double stage_exposure_before = learner->StageExposureSum();
            learner->ShrinkEpsilon(epsilon);
            learner->ResetStageEvaluation();
            std::cout << "STAGE_RESET algorithm=" << label
                      << " solver_regret_l1_before=" << solver_regret_before
                      << " solver_regret_l1_after=" << learner->SolverRegretL1()
                      << " stage_original_l1_before=" << stage_original_before
                      << " stage_original_l1_after=" << learner->StageOriginalRegretL1()
                      << " stage_W_sum_before=" << stage_exposure_before
                      << " stage_W_sum_after=" << learner->StageExposureSum() << '\n';
            ++stage_k;
            stage_t = 0;
            controller.ResetStage();
            policy_writer.Start(stage_k, epsilon);
            continue;
          }

          if (stage_timeout) {
            final_status = at_epsilon_min ? "FINAL_NOT_CERTIFIED" : "STAGE_TIMEOUT";
            policy_writer.Finish(false, stage_t);
            finished = true;
          } else if (global_limit) {
            final_status = at_epsilon_min ? "FINAL_NOT_CERTIFIED" : "MAX_GLOBAL_ITER";
            policy_writer.Finish(false, stage_t);
            finished = true;
          }
        }

        if (!finished) {
          const bool ended_at_epsilon_min =
              epsilon <= absl::GetFlag(FLAGS_epsilon_min) + std::numeric_limits<double>::epsilon() *
                                                                std::max(1.0, std::abs(epsilon));
          final_status = ended_at_epsilon_min ? "FINAL_NOT_CERTIFIED" : "MAX_GLOBAL_ITER";
          policy_writer.Finish(false, stage_t);
        }
        AdaptiveRecord run_end = last_record;
        run_end.record_type = "RUN_END";
        run_end.algorithm = label;
        run_end.global_t = global_t;
        run_end.stage_k = stage_k;
        run_end.stage_t = stage_t;
        run_end.epsilon = epsilon;
        run_end.epsilon_changed = false;
        run_end.quality_anchor_updated = false;
        run_end.stage_certified = final_status == "CERTIFIED_FINAL";
        run_end.status = final_status;
        adaptive_records[alg].push_back(run_end);
        run_end_seen[alg] = true;
        std::cout << "RUN_END algorithm=" << label << " status=" << final_status
                  << " final_stage=" << stage_k << " final_epsilon=" << epsilon
                  << " global_t=" << global_t << '\n';
      } catch (const std::invalid_argument& error) {
        process_ok[alg] = false;
        AdaptiveRecord run_end;
        run_end.record_type = "RUN_END";
        run_end.algorithm = label;
        run_end.status = "NUMERICAL_ERROR";
        adaptive_records[alg].push_back(run_end);
        run_end_seen[alg] = true;
        std::cerr << "RUN_END algorithm=" << label
                  << " status=NUMERICAL_ERROR error=" << error.what() << '\n';
      } catch (const std::exception& error) {
        process_ok[alg] = false;
        AdaptiveRecord run_end;
        run_end.record_type = "RUN_END";
        run_end.algorithm = label;
        run_end.status = "PROCESS_ERROR";
        adaptive_records[alg].push_back(run_end);
        run_end_seen[alg] = true;
        std::cerr << "RUN_END algorithm=" << label << " status=PROCESS_ERROR error=" << error.what()
                  << '\n';
      }
    };

    if (num_threads > 1) {
      threads.emplace(task);
      if (threads.size() >= num_threads) {
        hr_edl::Wait(threads.front());
        threads.pop();
      }
    } else {
      task();
    }
  }
  hr_edl::Wait(threads);

  for (size_t alg = 0; alg < alg_labels.size(); ++alg) {
    if (!run_end_seen[alg]) {
      process_ok[alg] = false;
      std::cerr << "Missing RUN_END for algorithm " << alg_labels[alg] << '\n';
    }
  }

  fs::path temp_path(output_path);
  temp_path += ".tmp";
  try {
    if (!output_path.parent_path().empty()) fs::create_directories(output_path.parent_path());
  } catch (const std::exception& error) {
    std::cerr << "Failed to create output directory: " << error.what() << '\n';
    return false;
  }
  std::ofstream out(temp_path, std::ios::out | std::ios::trunc);
  if (!out) {
    std::cerr << "Cannot open temporary result file: " << temp_path << '\n';
    return false;
  }

  if (adaptive) {
    WriteAdaptiveCsv(out, adaptive_records);
  } else {
    out << "algorithm  (ev, milliseconds, max_full_phi_regrets_fus, "
           "max_full_phi_regret, max_phi_regret, sum_phi_regret)\n";
    for (int64_t t = 0; t < iterations; ++t) {
      out << "t = " << t << '\n';
      for (size_t alg = 0; alg < alg_labels.size(); ++alg) {
        const auto& r = fixed_records[alg][t];
        out << alg_labels[alg] << "  "
            << absl::StrFormat("(%g, %g, %g, %g, %g, %g)", r.ev, r.milliseconds, r.posr_numerator,
                               r.osr_numerator, r.csr, r.acsr)
            << '\n';
      }
      out << '\n';
    }
  }
  out.flush();
  if (!out.good()) {
    std::cerr << "Failed writing result file: " << temp_path << '\n';
    return false;
  }
  out.close();
  if (out.fail()) {
    std::cerr << "Failed closing result file: " << temp_path << '\n';
    return false;
  }
  std::error_code rename_error;
  fs::rename(temp_path, output_path, rename_error);
  if (rename_error) {
    std::cerr << "Failed to replace result file: " << rename_error.message() << '\n';
    return false;
  }
  return std::all_of(process_ok.begin(), process_ok.end(), [](int ok) { return ok != 0; });
}

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("Run fixed-epsilon PBCE or adaptive stagewise PEFR.");
  absl::ParseCommandLine(argc, argv);
  return run_experiment() ? 0 : 1;
}
