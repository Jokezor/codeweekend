import math
import random


# Objective function: Rastrigin function
def objective_function(x):
    return 10 * len(x) + sum([(xi**2 - 10 * math.cos(2 * math.pi * xi)) for xi in x])


# Neighbor function: small random change
def get_neighbor(x, step_size, max_x, min_x):
    neighbor = x[:]
    index = random.randint(0, len(x) - 1)
    neighbor[index] += random.uniform(-step_size, step_size)
    neighbor[index] = max(min_x, min(neighbor[index], max_x))
    return neighbor


def adjust_temperature_alpha(initial_temp, current_iteration, alpha):
    """
    Adjust the temperature using an exponential decay based on alpha.

    Parameters:
    - initial_temp: The initial temperature.
    - current_iteration: The current iteration number.
    - alpha: The cooling rate.

    Returns:
    - The adjusted temperature.
    """
    return initial_temp * (alpha**current_iteration)


# Simulated Annealing function
def simulated_annealing(
    objective,
    bounds,
    n_iterations,
    step_size,
    initial_temp,
    alpha,
    num_turns,
    max_x,
    min_x,
):

    global_best = None
    global_best_eval = float("-inf")
    global_scores = []

    for _ in range(num_turns):
        # Initial solution
        best = [random.uniform(bound[0], bound[1]) for bound in bounds]
        best_eval = objective(best)
        current, current_eval = best, best_eval
        scores = [best_eval]

        for i in range(n_iterations):
            # Decrease temperature
            temp = adjust_temperature_alpha(initial_temp, i, alpha)

            if temp <= 1e-10:
                break
            # Generate candidate solution
            candidate = get_neighbor(current, step_size, max_x, min_x)
            candidate_eval = objective(candidate)
            # Check if we should keep the new solution
            if candidate_eval > best_eval or random.random() < math.exp(
                (current_eval - candidate_eval) / temp
            ):
                current, current_eval = candidate, candidate_eval
                if candidate_eval > best_eval:
                    best, best_eval = candidate, candidate_eval
                    scores.append(best_eval)

            # Optional: print progress
            # if i % 1000 == 0:
            #     print(
            #         f"Iteration {i}, Temperature {temp:.3f}, Best Evaluation {best_eval:.5f}"
            #     )
            #
        if best_eval > global_best_eval:
            global_best_eval = best_eval
            global_best = best
            global_scores = scores

    return global_best, global_best_eval, global_scores


# Define problem domain
bounds = [(-5.0, 5.0) for _ in range(2)]  # for a 2-dimensional Rastrigin function
n_iterations = 10000
temp = 100
n_turns = 100
alpha = 0.9999
step_size = 0.1
max_x = 5.0
min_x = -5.0

# Perform the simulated annealing search
best, score, scores = simulated_annealing(
    objective_function,
    bounds,
    n_iterations,
    step_size,
    temp,
    alpha,
    n_turns,
    max_x,
    min_x,
)

print(f"Best Solution: {best}")
print(f"Best Score: {score}")
