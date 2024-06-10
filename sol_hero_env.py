import gym
from gym import spaces
from stable_baselines3 import PPO
import torch
from multiprocessing import Pool
import numpy as np
import redis
import json


def clamp(value, min_value, max_value):
    return int(max(min_value, min(value, max_value)))


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


class HeroEnv(gym.Env):
    def __init__(self, data):
        super(HeroEnv, self).__init__()
        self.data = data
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32
        )  # x_move, y_move, attack
        self.observation_space = spaces.Box(
            low=0,
            high=max(data["width"], data["height"]),
            shape=(len(data["monsters"]) * 3 + 7,),
            dtype=np.float32,
        )
        self.reset()

    def reset(self):
        self.hero = self.data["hero"]
        self.hero["power"] = self.hero["base_power"]
        self.hero["speed"] = self.hero["base_speed"]
        self.hero["range"] = self.hero["base_range"]
        self.x, self.y = self.data["start_x"], self.data["start_y"]
        self.monsters = [monster.copy() for monster in self.data["monsters"]]
        self.score = 0
        self.turns = 0
        self.exp = 0
        self.level = 0
        self.moves = []
        return self._get_state()

    def _get_state(self):
        state = [
            self.x,
            self.y,
            self.hero["power"],
            self.hero["speed"],
            self.hero["range"],
            self.score,
            self.turns,
        ]
        for monster in self.monsters:
            state.extend([monster["x"], monster["y"], monster["hp"]])
        return np.array(state, dtype=np.float32)

    def step(self, action):
        score = 0
        intrinsic_reward = 0
        action_type = np.argmax(action)
        if action_type == 0:  # Move
            move_x = (action[0] * 2 - 1) * self.hero["speed"]
            move_y = (action[1] * 2 - 1) * self.hero["speed"]
            new_x = clamp(self.x + move_x, 0, self.data["width"])

            moved = abs(new_x - self.x)

            assert moved <= self.hero["speed"]
            sign_move_y = 1 if move_y > 0 else -1

            move_y = sign_move_y * (min(move_y, self.hero["speed"] - moved))

            new_y = clamp(self.y + move_y, 0, self.data["height"])
            self.x, self.y = new_x, new_y
            self.moves.append({"type": "move", "target_x": self.x, "target_y": self.y})
        elif action_type == 2:  # Attack
            attacked_monster = False
            for ind, monster in enumerate(self.monsters):
                if (monster["x"] - self.x) ** 2 + (
                    monster["y"] - self.y
                ) ** 2 <= self.hero["range"] ** 2 and monster["hp"] > 0:
                    self.monsters[ind]["hp"] -= self.hero["power"]
                    attacked_monster = True
                    if monster["hp"] <= 0:
                        score = monster["gold"]
                        self.exp += monster["exp"]
                        intrinsic_reward += monster["exp"] * 0.25
                    self.moves.append({"type": "attack", "target_id": ind})
                    break
            if not attacked_monster:
                self.turns -= 1
            exp_needed = 1000 + (self.level + 1) * self.level * 50
            if self.exp >= exp_needed:
                self.level += 1
                self.exp -= exp_needed
                old_worth = self.hero["power"] + self.hero["range"] + self.hero["speed"]
                self.hero["power"] = int(
                    self.hero["base_power"]
                    * (1 + self.level * self.hero["level_power_coeff"] / 100)
                )
                self.hero["range"] = int(
                    self.hero["base_range"]
                    * (1 + self.level * self.hero["level_range_coeff"] / 100)
                )
                self.hero["speed"] = int(
                    self.hero["base_speed"]
                    * (1 + self.level * self.hero["level_speed_coeff"] / 100)
                )
                new_worth = self.hero["power"] + self.hero["range"] + self.hero["speed"]
                intrinsic_reward = 0.5 * (new_worth - old_worth)

        self.turns += 1
        done = self.turns >= self.data["num_turns"]
        self.score += score
        reward = score + intrinsic_reward
        state = self._get_state()
        return state, reward, done, {}


def run_rl(test):
    with open(f"{test}.json", "r") as f_data:
        data = json.load(f_data)

    env = HeroEnv(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO("MlpPolicy", env, verbose=1, device=device)

    max_score = 0
    epochs = 5
    for epoch in range(epochs):
        model.learn(total_timesteps=10000)
        obs = env.reset()
        score = 0
        while True:
            action, _states = model.predict(obs)
            obs, score, rewards, done, info = env.step(action)
            score += rewards
            if done:
                break
        output = {"moves": env.moves}
        if score >= max_score:
            # if update_best_score(test, score, output):
            with open(f"output_hero_{test}.json", "w") as f:
                f.write(json.dumps(output, indent=4))
            max_score = score
        print(f"Epoch {epoch + 1}/{epochs} - Score: {score}")

    print(f"Test {test}: Max Score {max_score}")

    # update_best_score(test, score)


def main():
    tests = [str(i).zfill(3) for i in range(1, 2)]

    with Pool(processes=6) as pool:
        pool.map(run_rl, tests)


if __name__ == "__main__":
    main()
