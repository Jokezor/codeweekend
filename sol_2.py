import json
import random
from scipy.spatial import KDTree
import numpy as np


def build_kd_tree(monsters):
    positions = [(monster["x"], monster["y"]) for monster in monsters]
    return KDTree(positions), positions


def find_closest_monster(hero_pos, kd_tree, positions, monsters, range_squared):
    distances, indices = kd_tree.query(hero_pos, k=len(monsters))
    for distance, idx in zip(distances, indices):
        if distance**2 <= range_squared and monsters[idx]["hp"] > 0:
            return idx
    return -1


def monster_in_range(hero_pos, kd_tree, positions, monsters, range_squared):
    closest_idx = find_closest_monster(
        hero_pos, kd_tree, positions, monsters, range_squared
    )
    return closest_idx + 1 if closest_idx != -1 else 0


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def randomized_greedy_walk(hero_pos, target_pos, speed, max_x, max_y):
    new_pos = [hero_pos[0], hero_pos[1]]
    closest_monster_pos = target_pos

    x_steps_needed = abs(hero_pos[0] - closest_monster_pos[0])
    x_rand_walk = clamp(random.randint(0, x_steps_needed), 0, speed)
    if hero_pos[0] > closest_monster_pos[0]:
        new_pos[0] -= x_rand_walk
    else:
        new_pos[0] += x_rand_walk

    y_steps_needed = abs(hero_pos[1] - closest_monster_pos[1])
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


def main():
    tests = [str(i).zfill(3) for i in range(26, 51)]

    for test in tests:
        with open(f"{test}.json", "r") as f_data:
            data = json.load(f_data)

        max_score = 0
        kd_tree, positions = build_kd_tree(data["monsters"])

        for _ in range(10):
            hero = data["hero"]
            hero["power"] = hero["base_power"]
            hero["speed"] = hero["base_speed"]
            hero["range"] = hero["base_range"]

            x, y = data["start_x"], data["start_y"]
            max_x, max_y = data["width"], data["height"]
            max_turns = data["num_turns"]
            monsters = data["monsters"]
            monsters_killed = []

            turns = 0
            score = 0
            exp = 0
            level = 0
            target = 0
            closest_idx = 0
            output = {"moves": []}
            hero_pos = (x, y)

            while turns < max_turns:
                exp_needed = 1000 + (level + 1) * level * 50
                if exp >= exp_needed:
                    level += 1
                    exp -= exp_needed
                    hero["power"] = int(
                        hero["base_power"]
                        * (1 + level * hero["level_power_coeff"] / 100)
                    )
                    hero["range"] = int(
                        hero["base_range"]
                        * (1 + level * hero["level_range_coeff"] / 100)
                    )
                    hero["speed"] = int(
                        hero["base_speed"]
                        * (1 + level * hero["level_speed_coeff"] / 100)
                    )

                range_squared = hero["range"] ** 2
                speed_squared = hero["speed"] ** 2

                if target:
                    output["moves"].append(
                        {"type": "attack", "target_id": int(target - 1)}
                    )
                    # print(hero["range"])
                    monsters[target - 1]["hp"] -= hero["power"]
                    if monsters[target - 1]["hp"] <= 0:
                        score += monsters[target - 1]["gold"]
                        exp += monsters[target - 1]["exp"]
                        monsters_killed.append(target)
                        closest_idx = 0
                        target = 0
                else:
                    target = monster_in_range(
                        hero_pos, kd_tree, positions, monsters, range_squared
                    )
                    if target:
                        # print(hero["range"])
                        output["moves"].append(
                            {"type": "attack", "target_id": int(target - 1)}
                        )
                        monsters[target - 1]["hp"] -= hero["power"]
                        if monsters[target - 1]["hp"] <= 0:
                            monsters_killed.append(target)
                            exp += monsters[target - 1]["exp"]
                            score += monsters[target - 1]["gold"]
                            closest_idx = 0
                            target = 0
                    else:
                        if not closest_idx:
                            closest_idx = (
                                find_closest_monster(
                                    hero_pos, kd_tree, positions, monsters, float("inf")
                                )
                                + 1
                            )
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
                with open(f"output_{test}.json", "w") as f:
                    f.write(json.dumps(output, indent=4))
                max_score = score

        print(max_score)


if __name__ == "__main__":
    main()
