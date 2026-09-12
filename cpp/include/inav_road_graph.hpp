#pragma once

#include <cmath>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <algorithm>
#include <memory>

namespace inav {

constexpr double ROAD_EARTH_RADIUS = 6371000.0;
constexpr double ROAD_PI = 3.14159265358979323846;

struct Point2D {
    double p_N{0.0}; // North (meters)
    double p_E{0.0}; // East (meters)

    Point2D() = default;
    Point2D(double n, double e) : p_N(n), p_E(e) {}

    double distance_to(const Point2D& other) const {
        double dn = p_N - other.p_N;
        double de = p_E - other.p_E;
        return std::sqrt(dn * dn + de * de);
    }
};

struct RoadNode {
    int64_t id{0};
    double lat{0.0};
    double lon{0.0};
    Point2D pos;
    std::vector<int64_t> outgoing_edge_ids;
    std::vector<int64_t> incoming_edge_ids;
};

struct RoadSegment {
    int64_t edge_id{0};
    int sub_idx{0};
    Point2D p1;
    Point2D p2;
    double length_m{0.0};
    double heading_rad{0.0}; // Clockwise from North [0, 2*pi)
    bool is_oneway{false};
    std::string name;
};

struct RoadEdge {
    int64_t id{0};
    int64_t start_node_id{0};
    int64_t end_node_id{0};
    std::string name;
    std::string highway_type;
    bool is_oneway{false};
    double speed_limit_mps{13.88}; // ~50 km/h default
    double total_length_m{0.0};
    std::vector<Point2D> polyline;
    std::vector<RoadSegment> segments;
};

class SpatialGridIndex {
public:
    explicit SpatialGridIndex(double cell_size_m = 50.0) : cell_size_(cell_size_m) {}

    void clear() {
        grid_.clear();
    }

    void insert(const RoadSegment& seg) {
        int64_t c_n_min = coord_to_cell(std::min(seg.p1.p_N, seg.p2.p_N));
        int64_t c_n_max = coord_to_cell(std::max(seg.p1.p_N, seg.p2.p_N));
        int64_t c_e_min = coord_to_cell(std::min(seg.p1.p_E, seg.p2.p_E));
        int64_t c_e_max = coord_to_cell(std::max(seg.p1.p_E, seg.p2.p_E));

        for (int64_t cn = c_n_min; cn <= c_n_max; ++cn) {
            for (int64_t ce = c_e_min; ce <= c_e_max; ++ce) {
                uint64_t key = cell_key(cn, ce);
                grid_[key].push_back(seg);
            }
        }
    }

    std::vector<RoadSegment> query_radius(double p_N, double p_E, double radius_m) const {
        int64_t c_n_min = coord_to_cell(p_N - radius_m);
        int64_t c_n_max = coord_to_cell(p_N + radius_m);
        int64_t c_e_min = coord_to_cell(p_E - radius_m);
        int64_t c_e_max = coord_to_cell(p_E + radius_m);

        std::vector<RoadSegment> results;
        std::unordered_set<uint64_t> seen;

        for (int64_t cn = c_n_min; cn <= c_n_max; ++cn) {
            for (int64_t ce = c_e_min; ce <= c_e_max; ++ce) {
                uint64_t key = cell_key(cn, ce);
                auto it = grid_.find(key);
                if (it != grid_.end()) {
                    for (const auto& seg : it->second) {
                        uint64_t uid = (static_cast<uint64_t>(seg.edge_id) << 16) ^ static_cast<uint64_t>(seg.sub_idx);
                        if (seen.find(uid) == seen.end()) {
                            seen.insert(uid);
                            results.push_back(seg);
                        }
                    }
                }
            }
        }
        return results;
    }

private:
    double cell_size_{50.0};
    std::unordered_map<uint64_t, std::vector<RoadSegment>> grid_;

    int64_t coord_to_cell(double coord) const {
        return static_cast<int64_t>(std::floor(coord / cell_size_));
    }

    uint64_t cell_key(int64_t cn, int64_t ce) const {
        uint32_t un = static_cast<uint32_t>(cn);
        uint32_t ue = static_cast<uint32_t>(ce);
        return (static_cast<uint64_t>(un) << 32) | static_cast<uint64_t>(ue);
    }
};

class RoadGraph {
public:
    RoadGraph() = default;

    void clear() {
        nodes_.clear();
        edges_.clear();
        spatial_index_.clear();
        ref_lat_ = 0.0;
        ref_lon_ = 0.0;
        is_ref_set_ = false;
        min_lat_ = 90.0;
        max_lat_ = -90.0;
        min_lon_ = 180.0;
        max_lon_ = -180.0;
    }

    void set_reference_origin(double lat, double lon) {
        ref_lat_ = lat;
        ref_lon_ = lon;
        is_ref_set_ = true;
    }

    bool is_reference_set() const { return is_ref_set_; }
    double ref_lat() const { return ref_lat_; }
    double ref_lon() const { return ref_lon_; }

    Point2D latlon_to_local(double lat, double lon) const {
        double d_lat = (lat - ref_lat_) * (ROAD_PI / 180.0);
        double d_lon = (lon - ref_lon_) * (ROAD_PI / 180.0);
        double ref_lat_rad = ref_lat_ * (ROAD_PI / 180.0);
        double p_N = ROAD_EARTH_RADIUS * d_lat;
        double p_E = ROAD_EARTH_RADIUS * d_lon * std::cos(ref_lat_rad);
        return Point2D(p_N, p_E);
    }

    void local_to_latlon(double p_N, double p_E, double& out_lat, double& out_lon) const {
        double ref_lat_rad = ref_lat_ * (ROAD_PI / 180.0);
        double d_lat = p_N / ROAD_EARTH_RADIUS;
        double d_lon = p_E / (ROAD_EARTH_RADIUS * std::cos(ref_lat_rad));
        out_lat = ref_lat_ + d_lat * (180.0 / ROAD_PI);
        out_lon = ref_lon_ + d_lon * (180.0 / ROAD_PI);
    }

    void add_node(int64_t node_id, double lat, double lon) {
        RoadNode node;
        node.id = node_id;
        node.lat = lat;
        node.lon = lon;
        if (is_ref_set_) {
            node.pos = latlon_to_local(lat, lon);
        }
        if (lat < min_lat_) min_lat_ = lat;
        if (lat > max_lat_) max_lat_ = lat;
        if (lon < min_lon_) min_lon_ = lon;
        if (lon > max_lon_) max_lon_ = lon;
        nodes_[node_id] = node;
    }

    bool is_in_bounds(double lat, double lon, double margin_deg = 0.01) const {
        if (edges_.empty()) return false;
        return (lat >= min_lat_ - margin_deg && lat <= max_lat_ + margin_deg &&
                lon >= min_lon_ - margin_deg && lon <= max_lon_ + margin_deg);
    }

    bool is_local_in_bounds(double p_N, double p_E, double margin_deg = 0.01) const {
        if (!is_ref_set_ || edges_.empty()) return false;
        double lat = 0.0, lon = 0.0;
        local_to_latlon(p_N, p_E, lat, lon);
        return is_in_bounds(lat, lon, margin_deg);
    }

    void add_edge(
        int64_t edge_id,
        int64_t start_node_id,
        int64_t end_node_id,
        const std::string& name,
        const std::vector<Point2D>& polyline,
        bool is_oneway = false,
        double speed_limit_mps = 13.88,
        const std::string& highway_type = "residential"
    ) {
        if (polyline.size() < 2) return;

        RoadEdge edge;
        edge.id = edge_id;
        edge.start_node_id = start_node_id;
        edge.end_node_id = end_node_id;
        edge.name = name;
        edge.highway_type = highway_type;
        edge.is_oneway = is_oneway;
        edge.speed_limit_mps = speed_limit_mps;
        edge.polyline = polyline;
        edge.total_length_m = 0.0;

        for (size_t i = 0; i + 1 < polyline.size(); ++i) {
            const auto& p1 = polyline[i];
            const auto& p2 = polyline[i + 1];
            double d_N = p2.p_N - p1.p_N;
            double d_E = p2.p_E - p1.p_E;
            double len = std::sqrt(d_N * d_N + d_E * d_E);
            if (len < 1e-4) continue;

            // Heading: clockwise from North [0, 2*pi)
            double hdg = std::atan2(d_E, d_N);
            if (hdg < 0.0) hdg += 2.0 * ROAD_PI;

            RoadSegment seg;
            seg.edge_id = edge_id;
            seg.sub_idx = static_cast<int>(i);
            seg.p1 = p1;
            seg.p2 = p2;
            seg.length_m = len;
            seg.heading_rad = hdg;
            seg.is_oneway = is_oneway;
            seg.name = name;

            edge.segments.push_back(seg);
            edge.total_length_m += len;
            spatial_index_.insert(seg);
        }

        if (nodes_.find(start_node_id) != nodes_.end()) {
            nodes_[start_node_id].outgoing_edge_ids.push_back(edge_id);
        }
        if (nodes_.find(end_node_id) != nodes_.end()) {
            nodes_[end_node_id].incoming_edge_ids.push_back(edge_id);
            if (!is_oneway) {
                nodes_[end_node_id].outgoing_edge_ids.push_back(edge_id);
            }
        }

        edges_[edge_id] = edge;
    }

    const std::unordered_map<int64_t, RoadNode>& nodes() const { return nodes_; }
    const std::unordered_map<int64_t, RoadEdge>& edges() const { return edges_; }
    const SpatialGridIndex& spatial_index() const { return spatial_index_; }

    const RoadEdge* get_edge(int64_t edge_id) const {
        auto it = edges_.find(edge_id);
        if (it != edges_.end()) return &it->second;
        return nullptr;
    }

    bool are_edges_connected(int64_t e1_id, int64_t e2_id) const {
        if (e1_id == e2_id) return true;
        const auto* e1 = get_edge(e1_id);
        const auto* e2 = get_edge(e2_id);
        if (!e1 || !e2) return false;

        if (e1->end_node_id == e2->start_node_id) return true;
        if (!e2->is_oneway && e1->end_node_id == e2->end_node_id) return true;
        if (!e1->is_oneway && e1->start_node_id == e2->start_node_id) return true;
        if (!e1->is_oneway && !e2->is_oneway && e1->start_node_id == e2->end_node_id) return true;

        return false;
    }

private:
    std::unordered_map<int64_t, RoadNode> nodes_;
    std::unordered_map<int64_t, RoadEdge> edges_;
    SpatialGridIndex spatial_index_{50.0};
    double ref_lat_{0.0};
    double ref_lon_{0.0};
    bool is_ref_set_{false};
    double min_lat_{90.0};
    double max_lat_{-90.0};
    double min_lon_{180.0};
    double max_lon_{-180.0};
};

} // namespace inav
