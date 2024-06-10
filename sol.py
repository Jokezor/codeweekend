import json
import random

"""
So, we should walk according to a random walk, attack any monster in range.
Keep the best approach found.


Feels like a random walk could be useful at first..
So need to:

1. Make basic version keeping the base level.
2. Check so the move is valid within the world.
3. Move towards the biggest cluster of monsters.

Basic version:

1. Check if any monster is in range, if so attack it until it dies.
2. If no monsters in range, move in a random direction (x_h - x_t)^2 + (y_h - y_t)^2 <= hero_speed

"""

# Lets start with killing monsters in range.

tests = [str(i).zfill(3) for i in range(1, 4)]


# O(N)
def monster_in_range(x, y, monsters):
    in_range = [
        ind
        for ind, monster in enumerate(monsters)
        if (monster["x"] - x) ** 2 + (monster["y"] - y) ** 2 <= hero["base_range"] ** 2
        and monster["hp"] > 0
    ]
    return in_range[0] + 1 if len(in_range) else 0


for test in tests:

    f_data = open(f"{test}.json", "r")
    data = json.load(f_data)

    # Sort monsters according to gold/hp?
    # First simply check if any monsters in range.

    max_score = 0

    # O(k)
    for i in range(100):
        hero = data["hero"]
        x = data["start_x"]
        y = data["start_y"]

        max_x = data["width"]
        max_y = data["height"]
        max_turns = data["num_turns"]
        monsters = data["monsters"]
        monsters_killed = []

        turns = 0
        score = 0
        exp = 0
        target = 0
        output = {}
        output["moves"] = []

        # O(n)
        while turns < max_turns:
            if target:
                output["moves"].append({"type": "attack", "target_id": target - 1})
                # Keep attacking on same target
                monsters[target - 1]["hp"] -= hero["base_power"]
                if monsters[target - 1]["hp"] <= 0:
                    score += monsters[target - 1]["gold"]
                    exp += monsters[target - 1]["exp"]
                    monsters_killed.append(target)
                    target = 0
            elif target := monster_in_range(x, y, monsters):
                output["moves"].append({"type": "attack", "target_id": target - 1})
                # Finding new target
                monsters[target - 1]["hp"] -= hero["base_power"]
                if monsters[target - 1]["hp"] <= 0:
                    monsters_killed.append(target)
                    exp += monsters[target - 1]["exp"]
                    score += monsters[target - 1]["gold"]
                    target = 0
            else:
                # Walk
                # Now we should look for a random walk.
                # Once the random walk has been done, can be improved my aiming it for
                # The cluster of monsters.

                # This is where the most improvement comes from.

                # Get random x in range.
                x_candidate = random.randint(0, max_x)
                y_candidate = random.randint(0, max_y)

                # k

                # O(N^2)
                while (x_candidate - x) ** 2 + (y_candidate - y) ** 2 > hero[
                    "base_speed"
                ] ** 2:
                    x_candidate = random.randint(0, max_x)
                    y_candidate = random.randint(0, max_y)

                x = x_candidate
                y = y_candidate

                output["moves"].append({"type": "move", "target_x": x, "target_y": y})
            turns += 1

        if score > max_score:
            with open(f"output_{test}.json", "w") as f:
                f.write(json.dumps(output, indent=4))
            max_score = score

        # print(score)

    print(max_score)
