import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from pyeasyga import pyeasyga
import random
import pickle

max_generations = 1000
max_runs = 10 # per hyperparameter configuration
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
    # usr_ratings['rating'] = usr_ratings['rating'] - usr_mean
    rat = np.full(n_items, usr_mean)
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

# you can tell that this gets run with probability per-individual,
# but we want probability per single gene
def mutate(individual):
    global c
    mutate_prob = c[2]

    probabilities = np.random.rand(len(individual))
    mutate_indices = np.argwhere(probabilities > mutate_prob)

    all_random = np.random.randint(1, high=5, size=len(individual)) 
    individual[mutate_indices] = all_random[mutate_indices]

    fix_representation(individual)


histories = []

# [population, crossover, mutation]
config = [
    [20, 0.6, 0.0],
    [20, 0.6, 0.1]
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

# __xx variables = internal inter-evolution cycle usage

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
__h = []
__e = 0

def end_of_x_train_times_loop_stats():
    global __h, histories, best_fit, c
    __h = np.array(__h)
    # pad until max_gen
    histories.append([c, np.mean(__h, axis=0)])

    print()
    print(f'For config {c}')
    print(f'Best fit individual: {best_fit}')
    print(f'Average epochs: {__epochs / 10}')
    print()
    print('-----')
    print()


def plot_results():
    global histories
    with open('histories.pkl', 'wb') as f:
        pickle.dump(histories, f)

    fig, ax = plt.subplots(len(histories))
    
    fig.suptitle('Training results')
    for i, entry in enumerate(histories):
        plt.ylabel('average fitness')
        (c, hist) = entry
        ax[i].plot(hist)
        ax[i].set_title(str(c))

    plt.show()
    input('Plotted results')

# i opened an issue for that on their github: https://github.com/remiomosowon/pyeasyga/issues/12
# if they fix it fast I'll integrate it
# or maybe I'll implement it for them. I don't know. Too lazy to write tests
# in the meantime, we do a huge workaround to support it
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

    fitness_hist.append(best_person)

    reason = None
    if best_person_count > max_best_person_count:
        reason = f'best person did not improve for {best_person_count} cycles'

    if reason is not None:
        early_stopped = True
        print(f'[Early Stopping] Epoch {epochs}, reason: {reason}')
        end_training(early_stopped=True)


def fitness(individual, data):
    global fitness_hist, _best_person, best_neigh_arrays, population, _population_counter
    # cur = np.array(individual) - np.mean(np.array(individual))
    correlations = [np.corrcoef(individual, x)[0, 1] for x in best_neigh_arrays]
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

def start_training(end_callback):
    global user_ratings, create_individual, crossover, mutate, fitness, max_generations, ga, c
    ga = pyeasyga.GeneticAlgorithm(user_ratings,
                                   population_size=c[0],
                                   generations=max_generations,
                                   crossover_probability=c[1],
                                   mutation_probability=1.0,
                                   elitism=False,
                                   maximise_fitness=True)
    ga.create_individual = create_individual
    ga.crossover_function = crossover
    ga.mutate_function = mutate
    ga.fitness_function = fitness
    # we could also define a custom selection function

    ga.run()
    end_callback()

def end_training(early_stopped=False):
    global best_fit, __h, __epochs, max_generations

    if not early_stopped:
        print(f'Completed maximum generations ({max_generations}) for this run')

    cur_best_fit = ga.best_individual()[0]
    if best_fit is None or cur_best_fit > best_fit:
        best_fit = cur_best_fit

    print(f'Current best fit: {cur_best_fit}')
    padded = np.zeros(max_generations)
    padded[:len(fitness_hist)] = fitness_hist[:]
    __h.append(padded)

    __epochs = __epochs + epochs


    next_c = next(config_generator, None)
    if next_c is None:
        plot_results()
        print('Finished training')
    else:
        start_training(end_training)


def config_gen():
    global c, best_fit, population, fitness_hist, best_person, best_person_count, max_best_person_count, prev_fitness, early_stopped, epochs
    global __h, __epochs, __e

    for _c in config:
        c = _c
        __h = []
        __epochs = 0
        best_fit = None
        print(f'CONFIG: {c}')

        population = c[0]
        for i in range(max_runs):
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

            yield c

        end_of_x_train_times_loop_stats()

    plot_results()


config_generator = config_gen()
next(config_generator)
start_training(end_training)
