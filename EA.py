import random
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('team.csv')
data = data.sample(n=6, random_state=42).reset_index(drop=True)
teams = data['Team'].tolist()
venues = data['Stadium'].tolist()

times = ['10', '12', '14']
days = ['Friday', 'Saturday', 'Sunday']

swarm_size = 30
generations = 1000
random.seed(42)
fitness_history = []

def generate_schedule():
    match_count = int(len(teams) * (len(teams) - 1) / 2)
    individual = []
    seen = set()
    while len(individual) < match_count:
        t1, t2 = random.sample(teams, 2)
        pair = tuple(sorted([t1, t2]))
        if pair not in seen:
            seen.add(pair)
            match = [t1, t2, random.choice(venues), random.choice(days), random.choice(times), 0]
            individual.append(match)
    return individual

def assign_rounds(schedule):
    rounds = len(teams) - 1
    per_round = len(schedule) // rounds
    for i, match in enumerate(schedule):
        match[-1] = min((i // per_round) + 1, rounds)
    return schedule

def fitness(schedule):
    penalty = 0
    seen_pairs = set()
    round_match = {r: set() for r in range(1, len(teams))}
    slot_set = set()
    round_team = [set() for _ in range(len(teams))]
    for match in schedule:
        t1, t2, venue, day, time, rnd = match
        pair = tuple(sorted([t1, t2]))
        slot = (venue, day, time)
        if pair in seen_pairs: penalty += 1
        seen_pairs.add(pair)
        if pair in round_match[rnd]: penalty += 1
        round_match[rnd].add(pair)
        if slot in slot_set: penalty += 1
        slot_set.add(slot)
        if t1 in round_team[rnd]: penalty += 1
        if t2 in round_team[rnd]: penalty += 1
        round_team[rnd].update([t1, t2])
    return penalty

def generate_velocity(length):
    return [tuple(random.sample(range(length), 2)) for _ in range(5)]

def apply_velocity(schedule, velocity):
    schedule = schedule.copy()
    for i1, i2 in velocity:
        if i1 < len(schedule) and i2 < len(schedule):
            schedule[i1], schedule[i2] = schedule[i2], schedule[i1]
    return schedule

class Particle:
    def __init__(self, schedule):
        self.position = schedule
        self.best_position = schedule
        self.velocity = []
        self.best_fitness = fitness(schedule)

def fix_conflicts(schedule):
    rounds = len(teams) - 1
    matches_per_round = len(teams) // 2
    all_pairs = set(tuple(sorted([t1, t2])) for t1 in teams for t2 in teams if t1 != t2)
    used_slots = set()
    used_pairs = set()
    fixed_schedule = []
    for rnd in range(1, rounds + 1):
        round_teams = set()
        round_matches = []
        while len(round_matches) < matches_per_round:
            remaining_pairs = list(all_pairs - used_pairs)
            random.shuffle(remaining_pairs)
            for pair in remaining_pairs:
                t1, t2 = pair
                if t1 not in round_teams and t2 not in round_teams:
                    while True:
                        venue = random.choice(venues)
                        day = random.choice(days)
                        time = random.choice(times)
                        slot = (venue, day, time)
                        if slot not in used_slots:
                            used_slots.add(slot)
                            break
                    match = [t1, t2, venue, day, time, rnd]
                    round_matches.append(match)
                    round_teams.update([t1, t2])
                    used_pairs.add(pair)
                    break
        fixed_schedule.extend(round_matches)
    return fixed_schedule

swarm = []
for _ in range(swarm_size):
    schedule = assign_rounds(generate_schedule())
    swarm.append(Particle(schedule))

global_best = min(swarm, key=lambda p: p.best_fitness)

for gen in range(1, generations + 1):
    for p in swarm:
        velocity = generate_velocity(len(p.position))
        new_position = apply_velocity(p.position, velocity)
        new_position = assign_rounds(new_position)
        new_fitness = fitness(new_position)
        if new_fitness < p.best_fitness:
            p.best_position = new_position
            p.best_fitness = new_fitness
        p.velocity = velocity
        p.position = new_position
    current_best = min(swarm, key=lambda p: p.best_fitness)
    if current_best.best_fitness < global_best.best_fitness:
        global_best = current_best
    fitness_history.append(global_best.best_fitness)
    print(f"Generation {gen} - Best Fitness: {global_best.best_fitness}")
    if global_best.best_fitness == 0:
        print("Optimal schedule found!")
        break

global_best_schedule = fix_conflicts(global_best.best_position)
fitness_history.append(fitness(global_best_schedule))
global_best_schedule.sort(key=lambda x: x[-1])

day_order = {'Friday': 0, 'Saturday': 1, 'Sunday': 2}

print("\nFinal Corrected Schedule:")
for rnd in range(1, len(teams)):
    round_matches = [m for m in global_best_schedule if m[-1] == rnd]
    round_matches.sort(key=lambda x: day_order[x[3]])
    print(f"\nRound {rnd}:")
    for i, match in enumerate(round_matches, 1):
        print(f"Match {i}: {match[0]} vs {match[1]} at {match[2]} Stadium on {match[3]} at {match[4]} o'clock")

plt.plot(fitness_history, label="Best Fitness")
plt.title("Fitness Over Generations (PSO)")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid(True)
plt.legend()
plt.show()
