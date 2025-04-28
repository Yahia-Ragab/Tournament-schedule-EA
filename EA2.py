import random
import matplotlib.pyplot as plt
import seaborn as sns

teams = ['Arsenal', 'Real madrid', 'Chelsea', 'Barcelona', 'Byarn Munchen', 'Liverpool']
times = ['10', '12', '14']
days = ['Friday', 'Saturday', 'Sunday']
venues = ['Old trafford', 'Sintiago burnabio', 'Villa park']
fitness_history = []

pop_size = 200
gen = 1000

random.seed(42)

def generate_prnt():
    pop = []
    while len(pop) < pop_size:
        individual = []
        occure = set()
        n = float(len(teams))
        while len(individual) < (n * (n - 1) // 2):
            team1 = random.choice(teams)
            team2 = random.choice(teams)
            if team1 != team2 and tuple(sorted([team1, team2])) not in occure:
                occure.add(tuple(sorted([team1, team2])))
                match = [team1, team2, random.choice(venues), random.choice(days), random.choice(times)]
                individual.append(match)
        pop.append(individual)
    return pop

def set_rounds(population):
    round_number = len(teams) - 1
    for individual in population:
        matches_per_round = len(individual) // round_number
        for i, match in enumerate(individual):
            round_index = (i // matches_per_round) + 1
            round_index = min(round_index, round_number)
            match.append(round_index)
    return population

def fitness(schedule):
    penalty = 0
    for chromo in schedule:
        combination = set()
        round_match_dict = {r: set() for r in range(1, len(teams))}
        venue_day_time_set = set()
        team_round_play_count = [set() for _ in range(len(teams))]
        for match in chromo:
            team1, team2, venue, day, time, round = match
            pair = tuple(sorted([team1, team2]))
            venue_slot = (venue, day, time)
            if pair in combination:
                penalty += 1
            else:
                combination.add(pair)
            if pair in round_match_dict[round]:
                penalty += 1
            else:
                round_match_dict[round].add(pair)
            if venue_slot in venue_day_time_set:
                penalty += 1
            else:
                venue_day_time_set.add(venue_slot)
            if team1 in team_round_play_count[round]:
                penalty += 1
            team_round_play_count[round].add(team1)
            if team2 in team_round_play_count[round]:
                penalty += 1
            team_round_play_count[round].add(team2)

    return penalty

def select(population, tournament_size=25):
    selected_parents = []
    while len(selected_parents) < 2:
        tournament = random.sample(population, tournament_size)
        best_individual = min(tournament, key=lambda ind: fitness([ind]))
        selected_parents.append(best_individual)
    return selected_parents

def pos_select(population, k=25):
    # POS Selection (Optimistic selection)
    parents = []
    while len(parents) < 2:
        sample = random.sample(population, k)
        sample_sorted = sorted(sample, key=lambda ind: fitness([ind]))
        if random.random() < 0.8:
            parents.append(sample_sorted[0])  # Best with 80% chance
        else:
            parents.append(random.choice(sample_sorted[1:]))  # Otherwise from others
    return parents

def crossover(parent1, parent2):
    split_point1 = int(len(parent1) * 0.3)
    split_point2 = int(len(parent2) * 0.7)
    child1 = parent1[:split_point1] + parent2[split_point2:]
    child2 = parent2[:split_point1] + parent1[split_point2:]
    child1 = [match[:5] for match in child1]
    child2 = [match[:5] for match in child2]
    return child1, child2

def mutation(individual):
    if len(individual) < 3:
        return individual
    point1, point2 = random.sample(range(len(individual)), 2)
    individual[point1], individual[point2] = individual[point2], individual[point1]
    return individual

def generate_pop(prnt1, prnt2):
    new_population = []
    num_matches = int(len(teams) * (len(teams) - 1) // 2)
    while len(new_population) < pop_size:
        child1, child2 = crossover(prnt1, prnt2)
        child1 = child1[:num_matches]
        child2 = child2[:num_matches]
        occure1 = set(tuple(sorted([match[0], match[1]])) for match in child1)
        occure2 = set(tuple(sorted([match[0], match[1]])) for match in child2)
        while len(child1) < num_matches:
            team1, team2 = random.sample(teams, 2)
            pair = tuple(sorted([team1, team2]))
            if pair not in occure1:
                occure1.add(pair)
                match = [team1, team2, random.choice(venues), random.choice(days), random.choice(times)]
                child1.append(match)
        while len(child2) < num_matches:
            team1, team2 = random.sample(teams, 2)
            pair = tuple(sorted([team1, team2]))
            if pair not in occure2:
                occure2.add(pair)
                match = [team1, team2, random.choice(venues), random.choice(days), random.choice(times)]
                child2.append(match)
        new_population.extend([child1, child2])
    return new_population[:pop_size]


pop = generate_prnt()
pop = set_rounds(pop)

best_individual = min(pop, key=lambda ind: fitness([ind]))
best_fitness = fitness([best_individual])

global_best_individual = best_individual
global_best_fitness = best_fitness

for generation in range(1, gen + 1):
    s = pos_select(pop)  
    s1 = s[0]
    s2 = s[1]
    s1 = mutation(s1)
    s2 = mutation(s2)
    pop = generate_pop(s1, s2)
    pop = set_rounds(pop)

    best_individual = min(pop, key=lambda ind: fitness([ind]))
    best_fitness = fitness([best_individual])
    fitness_history.append(best_fitness)
    print(f"Generation {generation} - Best Fitness: {best_fitness}")
    if best_fitness < global_best_fitness:
        global_best_individual = best_individual
        global_best_fitness = best_fitness
    if global_best_fitness == 0:
        break

global_best_individual = sorted(global_best_individual, key=lambda match: match[-1])

print("\nBest Schedule Across All Generations:")
for match in global_best_individual:
    print(match)

print("\nBest Fitness Across All Generations:", global_best_fitness)


plt.plot(fitness_history, label='Fitness History', color='blue')
plt.title('Best Fitness History Over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.grid()
plt.show()
