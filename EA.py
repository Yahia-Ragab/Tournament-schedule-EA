import random
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
import copy

data = pd.read_csv('team.csv')
data = data.sample(n=6).reset_index(drop=True)
teams = data['Team'].tolist()
venues = data['Stadium'].tolist()
times = ['10', '12', '14', '16']
days = ['Friday', 'Saturday', 'Sunday']

population_size = 80
generations = 3000
mutation_rate = 0.3
crossover_rate = 0.8
elitism_rate = 0.1
stagnation_limit = 80
random.seed(42)
fitness_history = []

def generate_schedule():
    rounds = len(teams) - 1
    matches_per_round = len(teams) // 2
    match_pool = list(combinations(teams, 2))
    random.shuffle(match_pool)

    round_matches = [[] for _ in range(rounds)]
    team_used_in_round = [set() for _ in range(rounds)]

    for t1, t2 in match_pool:
        for r in range(rounds):
            if t1 not in team_used_in_round[r] and t2 not in team_used_in_round[r] and len(round_matches[r]) < matches_per_round:
                round_matches[r].append([t1, t2])
                team_used_in_round[r].update([t1, t2])
                break

    schedule = []
    used_slots = set()
    for r in range(rounds):
        for t1, t2 in round_matches[r]:
            for _ in range(100):
                venue = random.choice(venues)
                day = random.choice(days)
                time = random.choice(times)
                slot = (venue, day, time)
                if slot not in used_slots:
                    used_slots.add(slot)
                    schedule.append([t1, t2, venue, day, time, r + 1])
                    break
    return schedule

def fitness(schedule):
    penalty = 0
    seen_pairs = set()
    round_match = {r: set() for r in range(1, len(teams))}
    slot_set = set()
    round_team = [set() for _ in range(len(teams))]
    team_day_by_round = {team: {} for team in teams}
    for match in schedule:
        t1, t2, venue, day, time, rnd = match
        pair = tuple(sorted([t1, t2]))
        slot = (venue, day, time)
        if pair in seen_pairs:
            penalty += 1
        seen_pairs.add(pair)
        if pair in round_match[rnd]:
            penalty += 1
        round_match[rnd].add(pair)
        if slot in slot_set:
            penalty += 1
        slot_set.add(slot)
        if t1 in round_team[rnd]:
            penalty += 1
        if t2 in round_team[rnd]:
            penalty += 1
        round_team[rnd].update([t1, t2])
        team_day_by_round[t1][rnd] = day
        team_day_by_round[t2][rnd] = day
    for team in teams:
        for r in range(1, len(teams) - 1):
            if r in team_day_by_round[team] and r + 1 in team_day_by_round[team]:
                if team_day_by_round[team][r] == team_day_by_round[team][r + 1]:
                    penalty += 1
    return penalty

def crossover(parent1, parent2):
    min_len = min(len(parent1), len(parent2))
    child = []
    for i in range(min_len):
        child.append(random.choice([parent1[i], parent2[i]]))
    return child + random.choice([parent1, parent2])[min_len:]

def mutate(schedule):
    idx1, idx2, idx3 = random.sample(range(len(schedule)), 3)
    for idx in [idx1, idx2, idx3]:
        t1, t2, _, _, _, rnd = schedule[idx]
        venue = random.choice(venues)
        day = random.choice(days)
        time = random.choice(times)
        schedule[idx] = [t1, t2, venue, day, time, rnd]
    return schedule

def selection(population):
    tournament_size = 7
    selected = sorted(random.sample(population, tournament_size), key=fitness)
    return selected[:2]

def elitism(population):
    elite_size = max(1, int(elitism_rate * len(population)))
    return sorted(population, key=fitness)[:elite_size]

def reassign_rounds(schedule):
    matches = [(m[0], m[1]) for m in schedule]
    rounds = len(teams) - 1
    matches_per_round = len(teams) // 2
    random.shuffle(matches)

    round_matches = [[] for _ in range(rounds)]
    team_used_in_round = [set() for _ in range(rounds)]

    for t1, t2 in matches:
        for r in range(rounds):
            if t1 not in team_used_in_round[r] and t2 not in team_used_in_round[r] and len(round_matches[r]) < matches_per_round:
                round_matches[r].append((t1, t2))
                team_used_in_round[r].update([t1, t2])
                break

    new_schedule = []
    used_slots = set()
    for r in range(rounds):
        for t1, t2 in round_matches[r]:
            for _ in range(100):
                venue = random.choice(venues)
                day = random.choice(days)
                time = random.choice(times)
                slot = (venue, day, time)
                if slot not in used_slots:
                    used_slots.add(slot)
                    new_schedule.append([t1, t2, venue, day, time, r + 1])
                    break
    return new_schedule

def pso_optimize(schedule, iterations=100, w=0.5, c1=1.0, c2=1.0):
    def encode(match):
        return [venues.index(match[2]), days.index(match[3]), times.index(match[4])]

    def decode(t1, t2, code, rnd):
        return [t1, t2, venues[code[0]], days[code[1]], times[code[2]], rnd]

    def clamp(val, max_val):
        return max(0, min(val, max_val))

    particle_positions = [encode(m) for m in schedule]
    velocities = [[0, 0, 0] for _ in particle_positions]
    personal_best = particle_positions[:]
    global_best = particle_positions[:]
    personal_best_schedule = copy.deepcopy(schedule)
    global_best_schedule = copy.deepcopy(schedule)
    personal_best_fitness = fitness(personal_best_schedule)
    global_best_fitness = personal_best_fitness

    for _ in range(iterations):
        new_schedule = []
        for i in range(len(schedule)):
            t1, t2, _, _, _, rnd = schedule[i]
            for d in range(3):
                r1, r2 = random.random(), random.random()
                velocities[i][d] = int(
                    w * velocities[i][d] +
                    c1 * r1 * (personal_best[i][d] - particle_positions[i][d]) +
                    c2 * r2 * (global_best[i][d] - particle_positions[i][d])
                )
                max_val = [len(venues)-1, len(days)-1, len(times)-1][d]
                particle_positions[i][d] = clamp(particle_positions[i][d] + velocities[i][d], max_val)
            new_match = decode(t1, t2, particle_positions[i], rnd)
            new_schedule.append(new_match)

        new_fitness = fitness(new_schedule)
        if new_fitness < personal_best_fitness:
            personal_best = [encode(m) for m in new_schedule]
            personal_best_schedule = copy.deepcopy(new_schedule)
            personal_best_fitness = new_fitness

        if new_fitness < global_best_fitness:
            global_best = [encode(m) for m in new_schedule]
            global_best_schedule = copy.deepcopy(new_schedule)
            global_best_fitness = new_fitness

    return global_best_schedule

def evolutionary_algorithm():
    population = [generate_schedule() for _ in range(population_size)]
    best_schedule = min(population, key=fitness)
    
    stagnation = 0
    for gen in range(generations):
        if stagnation >= stagnation_limit:
            population = [copy.deepcopy(best_schedule) for _ in range(population_size)]
            stagnation = 0
        new_population = elitism(population)
        while len(new_population) < population_size:
            parents = selection(population)
            if random.random() < crossover_rate:
                child = crossover(parents[0], parents[1])
            else:
                child = parents[0][:]
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)
        population = new_population
        current_best = min(population, key=fitness)
        if fitness(current_best) < fitness(best_schedule):
            best_schedule = current_best
            stagnation = 0
        else:
            stagnation += 1

        if stagnation == 40:
            best_schedule = pso_optimize(best_schedule, iterations=10)

        fitness_history.append(fitness(best_schedule))
        print(f"Gen {gen + 1} - Best Fitness: {fitness(best_schedule)}")
        if fitness(best_schedule) == 0:
            break
    return best_schedule

def normalize_schedule(schedule, teams):
    rounds = len(teams) - 1
    matches_per_round = len(teams) // 2

    match_dict = {}
    for m in schedule:
        pair = tuple(sorted([m[0], m[1]]))
        match_dict[pair] = m

    matches = list(match_dict.values())
    random.shuffle(matches)

    round_matches = [[] for _ in range(rounds)]
    used_teams = [set() for _ in range(rounds)]

    new_schedule = []
    used_slots = set()

    for t1, t2, _, _, _, _ in matches:
        for r in range(rounds):
            if t1 not in used_teams[r] and t2 not in used_teams[r] and len(round_matches[r]) < matches_per_round:
                used_teams[r].update([t1, t2])
                round_matches[r].append((t1, t2))
                break

    for r in range(rounds):
        for t1, t2 in round_matches[r]:
            for _ in range(100):
                venue = random.choice(venues)
                day = random.choice(days)
                time = random.choice(times)
                slot = (venue, day, time)
                if slot not in used_slots:
                    used_slots.add(slot)
                    new_schedule.append([t1, t2, venue, day, time, r + 1])
                    break
    return new_schedule

final_schedule = evolutionary_algorithm()
final_schedule = normalize_schedule(final_schedule, teams)

day_order = {'Friday': 0, 'Saturday': 1, 'Sunday': 2}
final_schedule.sort(key=lambda x: x[-1])
for rnd in range(1, len(teams)):
    print(f"\nRound {rnd}:")
    for match in sorted([m for m in final_schedule if m[-1] == rnd], key=lambda x: day_order[x[3]]):
        print(f"{match[0]} vs {match[1]} at {match[2]} on {match[3]} at {match[4]}")

plt.plot(fitness_history, label="Best Fitness")
plt.title("Fitness Over Generations (Hybrid GA + PSO)")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid(True)
plt.legend()
plt.show()

from PyQt5.QtWidgets import QApplication, QWidget, QScrollArea, QVBoxLayout, QLabel, QGridLayout
from PyQt5.QtCore import Qt
import sys

from PyQt5.QtWidgets import (
    QApplication, QWidget, QScrollArea, QVBoxLayout, QLabel, QGridLayout, 
    QComboBox, QLineEdit, QHBoxLayout
)
from PyQt5.QtCore import Qt
import sys

def show_schedule_gui(schedule, teams, days):
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Tournament Schedule")

    main_layout = QVBoxLayout(window)

    controls_layout = QHBoxLayout()
    rounds = len(teams) - 1
    round_filter = QComboBox()
    round_filter.addItem("All Rounds")
    for r in range(1, rounds + 1):
        round_filter.addItem(f"Round {r}")
    controls_layout.addWidget(round_filter)

    search_input = QLineEdit()
    search_input.setPlaceholderText("Search by team name...")
    controls_layout.addWidget(search_input)
    main_layout.addLayout(controls_layout)

    scroll = QScrollArea()
    container = QWidget()
    grid = QGridLayout(container)

    cell_width = 180
    cell_height = 200

    for col in range(1, rounds + 1):
        label = QLabel(f"<b>Round {col}</b>")
        label.setAlignment(Qt.AlignCenter)
        label.setFixedSize(cell_width, 40)
        grid.addWidget(label, 0, col)

    for row, day in enumerate(days, start=1):
        label = QLabel(f"<b>{day}</b>")
        label.setAlignment(Qt.AlignCenter)
        label.setFixedSize(cell_width, 40)
        grid.addWidget(label, row, 0)

    cell_labels = {}
    for r in range(1, rounds + 1):
        for row, day in enumerate(days, start=1):
            label = QLabel()
            label.setAlignment(Qt.AlignTop)
            label.setWordWrap(True)
            label.setFixedSize(cell_width, cell_height)
            label.setStyleSheet("border: 1px solid gray; padding: 5px;")
            grid.addWidget(label, row, r)
            cell_labels[(r, day)] = label

    scroll.setWidget(container)
    scroll.setWidgetResizable(True)
    main_layout.addWidget(scroll)

    def update_display():
        selected_round_text = round_filter.currentText()
        selected_round = None
        if selected_round_text != "All Rounds":
            selected_round = int(selected_round_text.split()[-1])
        search_text = search_input.text().lower()

        for (r, day), label in cell_labels.items():
            if selected_round and r != selected_round:
                label.setText("")
                continue

            # Filter matches for this cell
            matches = [m for m in schedule if m[-1] == r and m[3] == day]
            if search_text:
                matches = [m for m in matches if search_text in m[0].lower() or search_text in m[1].lower()]
            if matches:
                text = "\n\n".join([f"{m[0]} vs {m[1]}\n{m[4]}:00\n{m[2]}" for m in matches])
            else:
                text = ""
            label.setText(text)

    round_filter.currentIndexChanged.connect(update_display)
    search_input.textChanged.connect(update_display)

    update_display()

    window.resize(1000, 600)
    window.show()
    sys.exit(app.exec_())

show_schedule_gui(final_schedule, teams, days)
