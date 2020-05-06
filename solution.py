import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from pyeasyga import pyeasyga
import random 

explore_hyperparams = False

# read to a dataframe
df = pd.read_csv('./u.data', sep='\t', header=None, usecols=[0, 1, 2])
df.columns = ['user_id', 'item_id', 'rating']
df['rating'] = df['rating'].values.astype(np.float32)

user_means = df.groupby(['user_id'])

user_ids = df['user_id'].unique()
item_ids = df['item_id'].unique()

n_users = len(user_ids)
n_items = len(item_ids)

cached_means = {}

print('Calculating user means')
for usr in user_ids:
    mean = user_means.get_group(usr)['rating'].mean()
    cached_means[usr] = mean

print('Sorting movie for output order')
sorted_item_ids = np.copy(item_ids)
np.sort(sorted_item_ids)
item_id_to_pos = {val: idx for idx, val in enumerate(sorted_item_ids)}

min_rating = min(df['rating'])
max_rating = max(df['rating'])

rand_user_id = df.sample(1).iloc[0]['user_id']
user_ratings = df.loc[df['user_id'] == rand_user_id][['item_id', 'rating']]

# pearson neighbourhood

# get the representation array that is going to be used for pearson correlation computation
def get_pearson_array_from_id(usr_id):
    usr_mean = cached_means[usr_id]
    usr_ratings = df[df['user_id'] == usr_id][['rating']]
    usr_ratings['rating'] = usr_ratings['rating'] - usr_mean
    rat = np.zeros(n_items)

    for _, row in user_ratings.iterrows():
        item_id = row.item_id
        rating = row.rating

        rat[item_id_to_pos[item_id]] = rating

    return rat

print("Calculating pearson correlations:")
selected_usr_arr = get_pearson_array_from_id(rand_user_id)
user_ids = np.delete(user_ids, np.where(user_ids == rand_user_id))

_a = [(x, get_pearson_array_from_id(x)) for x in user_ids]

best_neigh = [(x[0], np.corrcoef(selected_usr_arr, x[1])[0, 1], x[1]) for x in _a]
best_neigh.sort(key=lambda x: x[1], reverse=True)

print("Top 10 Neighborhood (user id, r):")
for x in best_neigh[:10]:
    print(f'({x[0]}, {x[1]})')

print()

best_neigh_arrays = [x[2] for x in best_neigh]

# force correct ratings for movies that are already rated
def fix_representation(representation):
    global user_ratings

    for _, row in user_ratings.iterrows():
        item_id = row.item_id
        rating = row.rating

        representation[item_id_to_pos[item_id]] = rating

def create_individual(ratings):
    representation = np.random.randint(1, high=5, size=n_items) 
    fix_representation(representation)
    return representation

def crossover(parent_1, parent_2):
    index = random.randrange(1, len(parent_1))
    child_1 = np.concatenate([parent_1[:index], parent_2[index:]])
    child_2 = np.concatenate([parent_2[:index], parent_1[index:]])
    
    # these should already be 'legal'
    return child_1, child_2

def mutate(individual):
    mutate_index = random.randrange(len(individual))
    mutate_val = random.randrange(1, 5)

    individual[mutate_index] = mutate_val
    # if we flip some already 
    fix_representation(individual)


histories = []

# [population, crossover, mutation]
config = [
    [20, 0.6, 0.001],
]

if explore_hyperparams:
    config = [
        [20, 0.6, 0.0],
        [20, 0.6, 0.01],
        [20, 0.6, 0.1],
        [20, 0.9, 0.01],
        [20, 0.1, 0.01],
        [200, 0.6, 0.0],
        [200, 0.6, 0.01],
        [200, 0.1, 0.01],
        [200, 0.9, 0.01],
    ]
# end if explore_hyperparams

# early stop parameters
best_person = None
best_person_count = 0

max_best_person_count = 2

prev_fitness = None

early_stopped = False
epochs = 0
# pyeasyga library does not support early stop callbacks 
# but we can hack it  because fitness function
# gets called every epoch

_best_person = None
fitness_hist = []

population = 0
_population_counter = 0

__e = 0
# i opened an issue for that on their github: https://github.com/remiomosowon/pyeasyga/issues/12
# if they fix it fast I'll integrate it
# or maybe I'll implement it for them. I don't know. Too lazy to write tests
ga = None
def early_stop_callback():
    global epochs, prev_fitness, early_stopped, best_person, best_person_count, max_best_person_count, ga, _best_person, __e

    __e = __e + 1
    if (__e % 20 == 0):
        print(f'Epoch: {__e}')
        print(f'Best fit: {best_person}')
    if early_stopped:
        return
    
    epochs = epochs + 1

    # prev_fitness should go here but the library does not expose
    # all the individuals (at least on the docs) so we dont implement it

    if best_person is None:
        best_person = _best_person

    if _best_person <= best_person:
        best_person_count = best_person_count + 1
    else:
        best_person_count = 0
        best_person = _best_person

    reason = None
    if best_person_count > max_best_person_count:
        reason = f'best person did not improve for {best_person_count} cycles'

    if reason is not None:
        early_stopped = True
        print(f'[Early Stopping] Epoch {epochs}, reason: {reason}')


def fitness(individual, data):
    global fitness_hist, _best_person, best_neigh_arrays, population, _population_counter
    cur = np.array(individual) - np.mean(np.array(individual))
    
    correlations = [np.corrcoef(cur, x)[0, 1] for x in best_neigh_arrays]
    fitness = np.mean(correlations)

    if _best_person is None:
        _best_person = fitness

    if fitness > _best_person:
        _best_person = fitness

    _population_counter = _population_counter + 1

    # finished an epoch
    if _population_counter % population == 0:
        early_stop_callback()
    
    return fitness

print('Starting training')
for c in config:
    __h = []
    __epochs = 0
    best_fit = None
    print(f'CONFIG: {c}')

    population = c[0]
    for i in range(10):
        print(f'    run: {i}')
        _population_counter = 0
        fitness_hist = []
        best_person = None
        best_person_count = 0

        max_best_person_count = 10

        prev_fitness = None
        early_stopped = False

        epochs = 0
        __e = 0

        ga = pyeasyga.GeneticAlgorithm(user_ratings,
                                    population_size=c[0],
                                    generations=1000,
                                    crossover_probability=c[1],
                                    mutation_probability=c[2],
                                    elitism=True,
                                    maximise_fitness=True)
        ga.create_individual = create_individual
        ga.crossover_function = crossover
        ga.mutate_function = mutate
        ga.fitness_function = fitness
        # we could also define a custom selection function

        ga.run()

        cur_best_fit = fitness(ga.best_individual(), [])
        if best_fit is None or cur_best_fit > best_fit:
            best_fit = cur_best_fit

        __h.append(np.array(copy.copy(fitness_hist)))
        __epochs = __epochs + epochs

    __h = np.array(__h)
    histories.append(copy.copy(x) for x in[c, np.mean(__h, axis=0)])

    print()
    print(f'For config {c}')
    print(f'Best fit individual: {best_fit}')
    print(f'Average epochs: {__epochs / 10}')


fig, ax = plt.subplots(len(histories))
fig.suptitle('Training results')

for i, entry in enumerate(histories):
    (c, hist) = entry
    ax[i].plot(hist)
    ax[i].xtitle(str(c))
    ax[i].ytitle('average fitness')
