#pragma once

#include "inav_road_graph.hpp"
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <limits>

namespace inav {

struct MapMatchResult {
    Point2D snapped_point;
    int64_t edge_id{0};
    int sub_idx{0};
    double confidence{0.0};
    double road_heading_rad{0.0};
    double cross_track_error_m{0.0};
    bool is_off_road{true};
    bool is_heading_valid{false};
};

struct TrellisCandidate {
    RoadSegment seg;
    Point2D proj;
    double d_perp{0.0};
    double log_score{0.0};
    int prev_idx{-1};
    double confidence{0.0};
};

class FixedLagHMMMapMatcher {
public:
    explicit FixedLagHMMMapMatcher(
        const RoadGraph* graph = nullptr,
        double sigma_road = 3.0,
        double beta = 5.0,
        double base_search_radius = 35.0,
        size_t max_lag_epochs = 50
    ) : graph_(graph),
        sigma_road_(sigma_road),
        beta_(beta),
        base_search_radius_(base_search_radius),
        max_lag_epochs_(max_lag_epochs) {}

    void set_graph(const RoadGraph* graph) {
        graph_ = graph;
        reset();
    }

    void reset() {
        prev_candidates_.clear();
        trellis_history_.clear();
    }

    static void project_point_to_segment(
        const Point2D& p,
        const Point2D& a,
        const Point2D& b,
        Point2D& out_proj,
        double& out_d_perp
    ) {
        double d_N = b.p_N - a.p_N;
        double d_E = b.p_E - a.p_E;
        double len_sq = d_N * d_N + d_E * d_E;

        if (len_sq < 1e-8) {
            out_proj = a;
            out_d_perp = p.distance_to(a);
            return;
        }

        double t = ((p.p_N - a.p_N) * d_N + (p.p_E - a.p_E) * d_E) / len_sq;
        t = std::clamp(t, 0.0, 1.0);
        out_proj.p_N = a.p_N + t * d_N;
        out_proj.p_E = a.p_E + t * d_E;
        out_d_perp = p.distance_to(out_proj);
    }

    static double wrap_to_pi(double angle_rad) {
        double a = std::fmod(angle_rad + ROAD_PI, 2.0 * ROAD_PI);
        if (a < 0.0) a += 2.0 * ROAD_PI;
        return a - ROAD_PI;
    }

    MapMatchResult match(
        double p_N,
        double p_E,
        double heading_rad,
        double speed_mps,
        double sigma_pos_m,
        double dt = 0.1
    ) {
        Point2D est_pt(p_N, p_E);

        if (!graph_ || graph_->edges().empty() || (graph_->is_reference_set() && !graph_->is_local_in_bounds(p_N, p_E))) {
            MapMatchResult res;
            res.snapped_point = est_pt;
            res.confidence = 0.0;
            res.road_heading_rad = heading_rad;
            res.is_off_road = true;
            return res;
        }

        // 1. Adaptive Search Radius based on estimator uncertainty
        double search_radius = std::clamp(3.0 * sigma_pos_m, base_search_radius_, 120.0);
        auto candidate_segs = graph_->spatial_index().query_radius(p_N, p_E, search_radius);

        if (candidate_segs.empty()) {
            prev_candidates_.clear();
            MapMatchResult res;
            res.snapped_point = est_pt;
            res.confidence = 0.0;
            res.road_heading_rad = heading_rad;
            res.is_off_road = true;
            return res;
        }

        // Limit candidate count to avoid exponential combinatorics
        if (candidate_segs.size() > 20) {
            std::sort(candidate_segs.begin(), candidate_segs.end(), [&](const RoadSegment& s1, const RoadSegment& s2) {
                Point2D pr1, pr2;
                double d1, d2;
                project_point_to_segment(est_pt, s1.p1, s1.p2, pr1, d1);
                project_point_to_segment(est_pt, s2.p1, s2.p2, pr2, d2);
                return d1 < d2;
            });
            candidate_segs.resize(20);
        }

        double travel_dist_m = std::max(speed_mps * dt, 0.05);
        double total_sigma_pos = std::sqrt(sigma_road_ * sigma_road_ + sigma_pos_m * sigma_pos_m);

        // Heading weight scaling based on speed
        double w_psi = 0.2;
        if (speed_mps > 12.0) w_psi = 2.0;
        else if (speed_mps > 3.0) w_psi = 1.0;

        std::vector<TrellisCandidate> current_candidates;
        current_candidates.reserve(candidate_segs.size());

        for (const auto& seg : candidate_segs) {
            TrellisCandidate cand;
            cand.seg = seg;
            project_point_to_segment(est_pt, seg.p1, seg.p2, cand.proj, cand.d_perp);

            // Emission Probability
            double log_p_d = -0.5 * std::pow(cand.d_perp / total_sigma_pos, 2);

            double d_psi = std::abs(wrap_to_pi(heading_rad - seg.heading_rad));
            if (!seg.is_oneway && d_psi > ROAD_PI / 2.0) {
                d_psi = std::abs(ROAD_PI - d_psi);
            }

            // Reverse travel on one-way road is strictly invalid
            if (seg.is_oneway && d_psi > ROAD_PI / 2.0) {
                continue;
            }

            double log_p_psi = -w_psi * d_psi;
            double log_emission = log_p_d + log_p_psi;

            // Transition Probability
            if (prev_candidates_.empty()) {
                cand.log_score = log_emission;
                cand.prev_idx = -1;
            } else {
                double max_prev_score = -std::numeric_limits<double>::infinity();
                int best_prev_idx = -1;

                for (size_t prev_i = 0; prev_i < prev_candidates_.size(); ++prev_i) {
                    const auto& prev = prev_candidates_[prev_i];
                    double dist_proj = cand.proj.distance_to(prev.proj);
                    double dist_diff = std::abs(dist_proj - travel_dist_m);
                    double log_trans = -dist_diff / beta_;

                    // Graph topological connectivity constraint
                    if (!graph_->are_edges_connected(prev.seg.edge_id, seg.edge_id)) {
                        log_trans -= 12.0; // Severe penalty for jumping disconnected roads
                    }

                    double total_score = prev.log_score + log_trans + log_emission;
                    if (total_score > max_prev_score) {
                        max_prev_score = total_score;
                        best_prev_idx = static_cast<int>(prev_i);
                    }
                }

                cand.log_score = max_prev_score;
                cand.prev_idx = best_prev_idx;
            }

            current_candidates.push_back(cand);
        }

        if (current_candidates.empty()) {
            prev_candidates_.clear();
            MapMatchResult res;
            res.snapped_point = est_pt;
            res.confidence = 0.0;
            res.road_heading_rad = heading_rad;
            res.is_off_road = true;
            return res;
        }

        // 2. Score Normalization across candidates to prevent numerical collapse
        double max_log = -std::numeric_limits<double>::infinity();
        for (const auto& c : current_candidates) {
            if (c.log_score > max_log) max_log = c.log_score;
        }

        double sum_exp = 0.0;
        for (auto& c : current_candidates) {
            c.log_score -= max_log; // Normalize so best is 0.0
            sum_exp += std::exp(c.log_score);
        }

        int best_idx = 0;
        double best_conf = 0.0;
        for (size_t i = 0; i < current_candidates.size(); ++i) {
            current_candidates[i].confidence = std::clamp(std::exp(current_candidates[i].log_score) / sum_exp, 0.0, 1.0);
            if (current_candidates[i].confidence > best_conf) {
                best_conf = current_candidates[i].confidence;
                best_idx = static_cast<int>(i);
            }
        }

        // 3. Fixed-Lag Trellis History Management
        trellis_history_.push_back(current_candidates);
        if (trellis_history_.size() > max_lag_epochs_) {
            trellis_history_.pop_front();
        }

        prev_candidates_ = current_candidates;

        const auto& best_cand = current_candidates[best_idx];

        MapMatchResult res;
        res.snapped_point = best_cand.proj;
        res.edge_id = best_cand.seg.edge_id;
        res.sub_idx = best_cand.seg.sub_idx;
        res.confidence = best_cand.confidence;
        res.cross_track_error_m = best_cand.d_perp;
        res.is_off_road = false;

        // Snapped heading alignment
        double aligned_heading = best_cand.seg.heading_rad;
        double d_hdg = std::abs(wrap_to_pi(heading_rad - aligned_heading));
        if (!best_cand.seg.is_oneway && d_hdg > ROAD_PI / 2.0) {
            aligned_heading = std::fmod(aligned_heading + ROAD_PI, 2.0 * ROAD_PI);
        }
        res.road_heading_rad = aligned_heading;
        res.is_heading_valid = (speed_mps > 2.5 && res.confidence > 0.5);

        return res;
    }

private:
    const RoadGraph* graph_{nullptr};
    double sigma_road_{3.0};
    double beta_{5.0};
    double base_search_radius_{35.0};
    size_t max_lag_epochs_{50};

    std::vector<TrellisCandidate> prev_candidates_;
    std::deque<std::vector<TrellisCandidate>> trellis_history_;
};

} // namespace inav
