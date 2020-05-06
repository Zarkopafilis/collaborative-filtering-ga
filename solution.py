import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

# mean centering for each user
df['rating'] = df.apply(lambda x: x['rating'] - cached_means[x['user_id']], axis=1)

# df.to_pickle('mean_centered_2.pkl')

# df['rating'] = df['rating'].subtract(df['rating'].mean())

min_rating = min(df['rating'])
max_rating = max(df['rating'])

# df.to_pickle('scaled_sigmoid_2.pkl')

print('Preparing user encoding')
user_id = int(row.user_id)
item_id = int(row.item_id)
user_mean = cached_means[user_id]

rating = row.rating
user_emb = usr_enc_map[user_id].toarray().squeeze()
_x = user_emb
# missing data goes here
_y = np.full(n_items, user_mean)  # fill with desired missing value / substitute

_y[item_id_to_pos[item_id]] = rating  # set correct item id rating


