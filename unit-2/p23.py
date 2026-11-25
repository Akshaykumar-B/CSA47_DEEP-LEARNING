"""Q-learning agent for choosing a green-light direction from traffic queues."""

from random import Random


random = Random(23)
queue_levels = range(5)
actions = (0, 1)  # 0: north-south green; 1: east-west green
q_values = {
    (north, east): [0.0, 0.0]
    for north in queue_levels
    for east in queue_levels
}
alpha, gamma, epsilon = 0.15, 0.9, 0.15


def state(queue):
    return min(queue[0], 4), min(queue[1], 4)


def step(queue, action, rng):
    next_queue = queue[:]
    # Each direction receives a random arrival; green can clear up to two vehicles.
    for direction in (0, 1):
        next_queue[direction] += rng.randint(0, 2)
    next_queue[action] = max(0, next_queue[action] - 2)
    next_queue = [min(value, 12) for value in next_queue]
    reward = -sum(next_queue)
    return next_queue, reward


for _ in range(700):
    queue = [random.randint(0, 4), random.randint(0, 4)]
    for _ in range(40):
        current_state = state(queue)
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = max(actions, key=lambda move: q_values[current_state][move])
        next_queue, reward = step(queue, action, random)
        next_state = state(next_queue)
        best_future = max(q_values[next_state])
        old_value = q_values[current_state][action]
        q_values[current_state][action] = old_value + alpha * (
            reward + gamma * best_future - old_value
        )
        queue = next_queue


def evaluate(policy, seed):
    rng = Random(seed)
    total_wait = 0
    queue = [2, 2]
    for time in range(1000):
        action = policy(queue, time)
        queue, reward = step(queue, action, rng)
        total_wait -= reward
    return total_wait / 1000


learned_wait = evaluate(lambda queue, _: max(actions, key=lambda a: q_values[state(queue)][a]), 230)
fixed_wait = evaluate(lambda _queue, time: (time // 2) % 2, 230)
print(f"Average queue per time step, learned policy: {learned_wait:.3f}")
print(f"Average queue per time step, fixed alternating policy: {fixed_wait:.3f}")
print("The agent learns from queue-based states and negative queue-length rewards.")
print("This is a small simulation; real traffic control needs safe constraints and calibrated data.")
