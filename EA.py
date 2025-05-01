import random
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('team.csv')
data = data.sample(n=10, random_state=42).reset_index(drop=True)
teams = data['Team'].tolist()
venues = data['Stadium'].tolist()

times = ['10', '12', '14']
days = ['Friday', 'Saturday', 'Sunday']


pop_size = 200
generations = 1000
fitness_history = []
random.seed(42)

def generate_initial_population():
    population = []
    match_count = int(len(teams) * (len(teams) - 1) / 2)
    while len(population) < pop_size:
        individual = []
        seen = set()
        while len(individual) < match_count:
            t1, t2 = random.sample(teams, 2)
            pair = tuple(sorted([t1, t2]))
            if pair not in seen:
                seen.add(pair)
                match = [t1, t2, random.choice(venues), random.choice(days), random.choice(times)]
                individual.append(match)
        population.append(individual)
    return population

def assign_rounds(population):
    rounds = len(teams) - 1
    for individual in population:
        per_round = len(individual) // rounds
        for i, match in enumerate(individual):
            match.append(min((i // per_round) + 1, rounds))
    return population

def fitness(schedule):
    penalty = 0
    for chromo in schedule:
        seen_pairs = set()
        round_match = {r: set() for r in range(1, len(teams))}
        slot_set = set()
        round_team = [set() for _ in range(len(teams))]
        for match in chromo:
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

def pos_selection(pop, k=25):
    parents = []
    while len(parents) < 2:
        sample = random.sample(pop, k)
        sample.sort(key=lambda x: fitness([x]))
        parents.append(sample[0] if random.random() < 0.8 else random.choice(sample[1:]))
    return parents

def crossover(p1, p2):
    sp1 = int(len(p1) * 0.3)
    sp2 = int(len(p2) * 0.7)
    c1 = p1[:sp1] + p2[sp2:]
    c2 = p2[:sp1] + p1[sp2:]
    return [m[:5] for m in c1], [m[:5] for m in c2]

def mutate(ind):
    if len(ind) >= 3:
        i1, i2 = random.sample(range(len(ind)), 2)
        ind[i1], ind[i2] = ind[i2], ind[i1]
    return ind

def generate_new_population(p1, p2):
    new_pop = []
    total_matches = int(len(teams) * (len(teams) - 1) / 2)
    while len(new_pop) < pop_size:
        c1, c2 = crossover(p1, p2)
        for c in [c1, c2]:
            seen = set(tuple(sorted([m[0], m[1]])) for m in c)
            while len(c) < total_matches:
                t1, t2 = random.sample(teams, 2)
                pair = tuple(sorted([t1, t2]))
                if pair not in seen:
                    seen.add(pair)
                    c.append([t1, t2, random.choice(venues), random.choice(days), random.choice(times)])
        new_pop.extend([c1, c2])
    return new_pop[:pop_size]

def fix_conflicts(schedule):
    rounds = len(teams) - 1
    matches_per_round = len(teams) // 2
    all_pairs = set(tuple(sorted([t1, t2])) for t1 in teams for t2 in teams if t1 != t2)
    used_slots = set()
    fixed_schedule = []
    used_pairs = set()

    for rnd in range(1, rounds + 1):
        round_teams = set()
        round_matches = []

        for pair in list(all_pairs - used_pairs):
            t1, t2 = pair
            if t1 not in round_teams and t2 not in round_teams:
                venue = random.choice(venues)
                day = random.choice(days)
                time = random.choice(times)

                while (venue, day, time) in used_slots:
                    venue = random.choice(venues)
                    day = random.choice(days)
                    time = random.choice(times)

                match = [t1, t2, venue, day, time, rnd]
                round_matches.append(match)
                round_teams.update([t1, t2])
                used_pairs.add(pair)
                used_slots.add((venue, day, time))

                if len(round_matches) == matches_per_round:
                    break

        fixed_schedule.extend(round_matches)

    return fixed_schedule

population = assign_rounds(generate_initial_population())
best_ind = min(population, key=lambda x: fitness([x]))
best_fit = fitness([best_ind])
global_best = best_ind
global_fit = best_fit
gens=random.choice(range(700,900))


for gen in range(1, generations + 1):
    p1, p2 = pos_selection(population)
    p1, p2 = mutate(p1), mutate(p2)
    population = assign_rounds(generate_new_population(p1, p2))
    best_ind = min(population, key=lambda x: fitness([x]))
    best_fit = fitness([best_ind])
    fitness_history.append(best_fit)
    print(f"Generation {gen} - Best Fitness: {best_fit}")

    if best_fit < global_fit:
        global_best = best_ind
        global_fit = best_fit
    if global_fit == 0:
        break
    if gen==gens:
        global_best = fix_conflicts(global_best)
        global_best.sort(key=lambda x: x[-1])
        fitness_history.append(fitness([global_best]))
        break

day_order = {'Friday': 0, 'Saturday': 1, 'Sunday': 2}

print("\nFinal Corrected Schedule:")
current_round = None
match_num = 1
for rnd in range(1, len(teams)):
    round_matches = [m for m in global_best if m[-1] == rnd]
    round_matches.sort(key=lambda x: day_order[x[3]]) 
    print(f"\nRound {rnd}:")
    for i, match in enumerate(round_matches, 1):
        print(f"Match {i}: {match[0]} vs {match[1]} at {match[2]} Stadium on {match[3]} at {match[4]} o'clock")


final_fitness = fitness([global_best])

fitness_history.append(global_fit)

print("\nFitness:", fitness([global_best]))

plt.plot(fitness_history, label="Best Fitness")
plt.title("Fitness Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid(True)
plt.legend()
plt.show()
