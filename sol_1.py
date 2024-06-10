import json
import random
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import asyncio
from multiprocessing import Pool
import numpy as np
import redis
from submit import submit, get_submission_info

# Initialize Redis client
redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)


def update_best_score(test, score, data):
    current_best = redis_client.get(test)
    if current_best is None or score > int(current_best):
        redis_client.set(test, score)
        print(f"New best score for {test}: {score}")
        return 1
    else:
        return 0


def get_best_score(test):
    score = redis_client.get(test)
    return int(score) if score else None


def build_kd_tree(monsters):
    positions = [(monster["x"], monster["y"]) for monster in monsters]

    if not monsters:
        return None, []

    print(positions)
    return KDTree(positions), positions


def find_closest_monster(hero_pos, kd_tree, positions, monsters, range_squared):
    distances, indices = kd_tree.query(hero_pos, k=len(monsters))
    closest = []

    if not isinstance(distances, np.ndarray):
        distances = np.array([distances])
        indices = np.array([indices])

    for distance, idx in zip(distances, indices):
        if distance**2 <= range_squared and monsters[idx]["hp"] > 0:
            closest.append(idx)
    return closest[random.randint(0, len(closest) - 1)] if closest else -1


# def monster_in_range(hero_pos, kd_tree, positions, monsters, range_squared):
#     closest_idx = find_closest_monster(
#         hero_pos, kd_tree, positions, monsters, range_squared
#     )
#     return closest_idx + 1 if closest_idx != -1 else 0
#


def score_monster(monster, weight_gold_hp=0.5, weight_exp_hp=0.5):
    hp = monster["hp"]
    score = 0
    if hp > 0:
        gold_hp_ratio = monster["gold"] / hp
        exp_hp_ratio = monster["exp"] / hp
        score = (weight_gold_hp * gold_hp_ratio) + (weight_exp_hp * exp_hp_ratio)
    return score


def find_top_scoring_monsters(
    hero_pos, kd_tree, positions, monsters, max_distance_squared, top_n=5, ratio=0.5
):
    distances, indices = kd_tree.query(hero_pos, k=len(monsters))
    scored_monsters = []

    if not isinstance(distances, np.ndarray):
        distances = np.array([distances])
        indices = np.array([indices])

    for distance, idx in zip(distances, indices):
        if distance**2 <= max_distance_squared and monsters[idx]["hp"] > 0:
            score = score_monster(monsters[idx], ratio, 1 - ratio)
            scored_monsters.append((score, idx))

    scored_monsters.sort(reverse=True, key=lambda x: x[0])
    return scored_monsters[:top_n]


def monster_in_range(hero_pos, kd_tree, positions, monsters, range_squared, ratio):
    top_monsters = find_top_scoring_monsters(
        hero_pos, kd_tree, positions, monsters, range_squared, top_n=1, ratio=ratio
    )
    return top_monsters[0][1] + 1 if top_monsters else 0


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def randomized_greedy_walk(hero_pos, target_pos, speed, max_x, max_y):
    new_pos = [hero_pos[0], hero_pos[1]]
    closest_monster_pos = target_pos

    x_steps_needed = abs(hero_pos[0] - closest_monster_pos[0])
    # x_rand_walk = clamp(x_steps_needed, 0, speed)
    x_rand_walk = clamp(random.randint(0, x_steps_needed), 0, speed)
    if hero_pos[0] > closest_monster_pos[0]:
        new_pos[0] -= x_rand_walk
    else:
        new_pos[0] += x_rand_walk

    y_steps_needed = abs(hero_pos[1] - closest_monster_pos[1])
    # y_rand_walk = clamp(y_steps_needed, 0, speed - x_rand_walk)
    y_rand_walk = clamp(random.randint(0, y_steps_needed), 0, speed - x_rand_walk)

    if x_rand_walk < speed:

        if hero_pos[1] > closest_monster_pos[1]:
            new_pos[1] -= y_rand_walk
        else:
            new_pos[1] += y_rand_walk

    return tuple(map(int, new_pos))


def level_up(level, exp):
    exp_needed = 1000 + level * (level - 1)

    return exp > exp_needed


def run_test(test):
    with open(f"{test}.json", "r") as f_data:
        data = json.load(f_data)

    max_score = 0
    monsters = [monster.copy() for monster in data["monsters"]]

    for _ in range(10):
        hero = data["hero"]
        hero["power"] = hero["base_power"]
        hero["speed"] = hero["base_speed"]
        hero["range"] = hero["base_range"]

        x, y = data["start_x"], data["start_y"]
        max_x, max_y = data["width"], data["height"]
        max_turns = data["num_turns"]
        monsters_alive = [monster for monster in monsters if monster["hp"] > 0]

        kd_tree, positions = build_kd_tree(monsters_alive)
        monsters_killed = []

        turns = 0
        score = 0
        exp = 0
        level = 0
        target = 0
        closest_idx = 0
        output = {"moves": []}
        hero_pos = (x, y)
        ratio = random.random()

        while turns < max_turns:
            exp_needed = 1000 + (level + 1) * level * 50
            if exp >= exp_needed:
                level += 1
                exp -= exp_needed
                hero["power"] = int(
                    hero["base_power"] * (1 + level * hero["level_power_coeff"] / 100)
                )
                hero["range"] = int(
                    hero["base_range"] * (1 + level * hero["level_range_coeff"] / 100)
                )
                hero["speed"] = int(
                    hero["base_speed"] * (1 + level * hero["level_speed_coeff"] / 100)
                )

            range_squared = hero["range"] ** 2
            speed_squared = hero["speed"] ** 2

            if target:
                output["moves"].append({"type": "attack", "target_id": int(target - 1)})
                monsters_alive[target - 1]["hp"] -= hero["power"]
                if monsters_alive[target - 1]["hp"] <= 0:
                    score += monsters_alive[target - 1]["gold"]
                    exp += monsters_alive[target - 1]["exp"]
                    monsters_killed.append(target)
                    monsters_alive.pop(target - 1)
                    kd_tree, positions = build_kd_tree(monsters_alive)
                    closest_idx = 0
                    target = 0
            else:
                target = monster_in_range(
                    hero_pos, kd_tree, positions, monsters_alive, range_squared, ratio
                )
                if target:
                    output["moves"].append(
                        {"type": "attack", "target_id": int(target - 1)}
                    )
                    monsters_alive[target - 1]["hp"] -= hero["power"]
                    if monsters_alive[target - 1]["hp"] <= 0:
                        monsters_killed.append(target)
                        exp += monsters_alive[target - 1]["exp"]
                        score += monsters_alive[target - 1]["gold"]
                        monsters_alive.pop(target - 1)
                        kd_tree, positions = build_kd_tree(monsters_alive)
                        closest_idx = 0
                        target = 0
                else:
                    if not closest_idx:
                        top_monsters = find_top_scoring_monsters(
                            hero_pos, kd_tree, positions, monsters_alive, float("inf")
                        )
                        if top_monsters:
                            closest_idx = (
                                top_monsters[random.randint(0, len(top_monsters) - 1)][
                                    1
                                ]
                                + 1
                            )
                            # closest_idx = (
                            #     find_closest_monster(
                            #         hero_pos, kd_tree, positions, monsters, float("inf")
                            #     )
                            #     + 1
                            # )
                    hero_pos = randomized_greedy_walk(
                        hero_pos,
                        positions[closest_idx - 1],
                        hero["speed"],
                        max_x,
                        max_y,
                    )
                    x, y = hero_pos
                    output["moves"].append(
                        {"type": "move", "target_x": int(x), "target_y": int(y)}
                    )
            turns += 1

        if score >= max_score:
            if update_best_score(test, score, output):
                with open(f"output_{test}.json", "w") as f:
                    f.write(json.dumps(output, indent=4))
            max_score = score


def main():
    tests = [str(i).zfill(3) for i in range(1, 6)]

    # To profile
    run_test(tests[1])

    # for i in range(100):
    #     with Pool(processes=5) as pool:
    #         pool.map(run_test, tests)
    #


if __name__ == "__main__":
    main()
