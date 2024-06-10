import gym
from gym import spaces
import numpy as np


class HeroEnv(gym.Env):
    def __init__(self, data):
        super(HeroEnv, self).__init__()
        self.data = data
        self.action_space = spaces.Discrete(5)  # Up, Down, Left, Right, Attack
        self.observation_space = spaces.Box(
            low=0,
            high=max(data["width"], data["height"]),
            shape=(len(data["monsters"]) * 3 + 4,),
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
        return self._get_state()

    def _get_state(self):
        state = [
            self.x,
            self.y,
            self.hero["power"],
            self.hero["speed"],
            self.hero["range"],
        ]
        for monster in self.monsters:
            state.extend([monster["x"], monster["y"], monster["hp"]])
        return np.array(state, dtype=np.float32)

    def step(self, action):
        if action == 0:  # Up
            self.y = min(self.y + self.hero["speed"], self.data["height"])
        elif action == 1:  # Down
            self.y = max(self.y - self.hero["speed"], 0)
        elif action == 2:  # Left
            self.x = max(self.x - self.hero["speed"], 0)
        elif action == 3:  # Right
            self.x = min(self.x + self.hero["speed"], self.data["width"])
        elif action == 4:  # Attack
            for monster in self.monsters:
                if (monster["x"] - self.x) ** 2 + (
                    monster["y"] - self.y
                ) ** 2 <= self.hero["range"] ** 2 and monster["hp"] > 0:
                    monster["hp"] -= self.hero["power"]
                    if monster["hp"] <= 0:
                        self.score += monster["gold"]
                        self.exp += monster["exp"]
                        self.monsters.remove(monster)
                    break
        self.turns += 1
        done = self.turns >= self.data["num_turns"]
        reward = self.score
        return self._get_state(), reward, done, {}


def run_rl(test):
    with open(f"{test}.json", "r") as f_data:
        data = json.load(f_data)

    env = HeroEnv(data)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    score = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        score += rewards
        if done:
            break

    update_best_score(test, score)
    print(f"Test {test}: Max Score {score}")


def main():
    tests = [str(i).zfill(3) for i in range(1, 26)]
    with Pool(processes=8) as pool:
        pool.map(run_rl, tests)


if __name__ == "__main__":
    main()
