#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

struct Monster {
    double x, y;
    double hp, gold, exp;
};

extern "C" {

    void build_kd_tree(const double* monster_positions, int n, double* tree) {
        for (int i = 0; i < n; ++i) {
            tree[2*i] = monster_positions[2*i];
            tree[2*i+1] = monster_positions[2*i+1];
        }
    }

  void find_top_scoring_monsters(const double* hero_pos, const double* monster_positions, const double* hp, const double* gold, const double* exp, int n, double max_distance_squared, int top_n, double ratio, double* scores, int* indices) {
        double hero_x = hero_pos[0];
        double hero_y = hero_pos[1];
        std::vector<std::pair<double, int>> scored_monsters;

        for (int i = 0; i < n; ++i) {
            if (hp[i] <= 0) continue;
            double distance_squared = (hero_x - monster_positions[2*i]) * (hero_x - monster_positions[2*i]) + (hero_y - monster_positions[2*i+1]) * (hero_y - monster_positions[2*i+1]);
            if (distance_squared <= max_distance_squared) {
                double gold_hp_ratio = gold[i] / hp[i];
                double exp_hp_ratio = exp[i] / hp[i];
                double score = ratio * gold_hp_ratio + (1 - ratio) * exp_hp_ratio;
                scored_monsters.push_back(std::make_pair(score, i));
            }
        }

        std::sort(scored_monsters.rbegin(), scored_monsters.rend());

        for (int i = 0; i < top_n && i < scored_monsters.size(); ++i) {
            scores[i] = scored_monsters[i].first;
            indices[i] = scored_monsters[i].second;
        }
    }

    void find_closest_monster(const double* hero_pos, const double* monster_positions, const double* hp, int n, double range_squared, int* closest_idx) {
        double hero_x = hero_pos[0];
        double hero_y = hero_pos[1];
        double min_distance_squared = range_squared;
        int closest = -1;

        for (int i = 0; i < n; ++i) {
            if (hp[i] <= 0) continue;
            double distance_squared = (hero_x - monster_positions[2*i]) * (hero_x - monster_positions[2*i]) + (hero_y - monster_positions[2*i+1]) * (hero_y - monster_positions[2*i+1]);
            if (distance_squared <= min_distance_squared) {
                min_distance_squared = distance_squared;
                closest = i;
            }
        }

        *closest_idx = closest;
    }

}
