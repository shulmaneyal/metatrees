from os import path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FOLDER = path.dirname(path.realpath(__file__))


def get_movie_lens_100k(subset=True):
    def _get_ratings(ratings_file):
        ratings_file_path = path.join(DATA_FOLDER, 'ml-100k', ratings_file)
        items_file_path = path.join(DATA_FOLDER, 'ml-100k', 'u.item')
        users_file_path = path.join(DATA_FOLDER, 'ml-100k', 'u.user')

        ratings = pd.read_csv(ratings_file_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        users = pd.read_csv(users_file_path, sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
        users['male'] = users.gender == 'M'
        users = users.drop(['occupation', 'gender'], axis=1)

        items_columns = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url', 'unknown',
                         'action', 'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary', 'drama',
                         'fantasy', 'film-noir', 'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war',
                         'western']

        items = pd.read_csv(items_file_path, sep='|', encoding='latin-1', names=items_columns) \
            .assign(release_date=lambda x: pd.to_numeric(x.release_date.str[-4:]))
        items = items.drop(['video_release_date', 'unknown'], axis=1)
        items.release_date = items.release_date.fillna(items.release_date.median())

        ratings = ratings.merge(users, on='user_id').merge(items, on='item_id')
        ratings = ratings.drop(['zip_code', 'movie_title', 'imdb_url'], axis=1)

        return ratings.sort_values('timestamp').reset_index(drop=True).drop('timestamp', axis=1).astype('float')

    train = _get_ratings('u1.base')
    test = _get_ratings('u1.test')

    movie_means = train.groupby('item_id').rating.agg({'mean', 'count'}).rename(
        columns={'mean': 'movie_ratings_mean', 'count': 'movie_ratings_count'})
    average_rating = train.rating.mean()
    train = train.merge(movie_means, left_on='item_id', right_index=True)
    train.movie_ratings_mean = (
            (train.movie_ratings_mean * train.movie_ratings_count - train.rating) / (train.movie_ratings_count - 1))
    train.movie_ratings_mean = train.movie_ratings_mean.fillna(average_rating)
    train.movie_ratings_count -= 1

    test = test.merge(movie_means, left_on='item_id', right_index=True, how='left')
    test.movie_ratings_mean = test.movie_ratings_mean.fillna(average_rating)
    test.movie_ratings_count = test.movie_ratings_count.fillna(0)

    if subset:
        return train.drop('item_id', axis=1), test.drop('item_id', axis=1)
    return train, test


def get_movie_lens_1m(subset=True):
    items_file_path = path.join(DATA_FOLDER, 'ml-1m', 'movies.dat')
    users_file_path = path.join(DATA_FOLDER, 'ml-1m', 'users.dat')
    ratings_file_path = path.join(DATA_FOLDER, 'ml-1m', 'ratings.dat')

    items = pd.read_csv(items_file_path, sep='::', engine='python', names=['item_id', 'movie_title', 'genres'])
    items['release_date'] = items.movie_title.str[-5:-1].astype('float')
    genres = items.genres.str.get_dummies(sep='|').rename(columns=str.lower)
    items = items.join(genres).drop(['genres', 'movie_title'], axis=1)

    users = pd.read_csv(users_file_path, sep='::', engine='python',
                        names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])
    users['male'] = (users.gender == 'M').astype('float')
    users = users.drop(['gender', 'occupation', 'zip_code'], axis=1)

    ratings = pd.read_csv(ratings_file_path, sep='::', engine='python',
                          names=['user_id', 'item_id', 'rating', 'timestamp'])
    ratings = ratings.drop('timestamp', axis=1)
    ratings = ratings.merge(items, on='item_id')
    ratings = ratings.merge(users, on='user_id')

    train, test = train_test_split(ratings, test_size=0.1, random_state=0)

    movie_means = train.groupby('item_id').rating.agg({'mean', 'count'}).rename(
        columns={'mean': 'movie_ratings_mean', 'count': 'movie_ratings_count'})
    average_rating = train.rating.mean()
    train = train.merge(movie_means, left_on='item_id', right_index=True)
    train.movie_ratings_mean = (
            (train.movie_ratings_mean * train.movie_ratings_count - train.rating) / (train.movie_ratings_count - 1))
    train.movie_ratings_mean = train.movie_ratings_mean.fillna(average_rating)
    train.movie_ratings_count -= 1

    test = test.merge(movie_means, left_on='item_id', right_index=True, how='left')
    test.movie_ratings_count = test.movie_ratings_count.fillna(0)
    test.movie_ratings_mean = test.movie_ratings_mean.fillna(average_rating)

    if subset:
        return train.drop('item_id', axis=1), test.drop('item_id', axis=1)
    return train, test


def get_jester(subset=True):
    train_path = path.join(DATA_FOLDER, 'jester', 'train.csv')
    test_path = path.join(DATA_FOLDER, 'jester', 'test.csv')
    items_path = path.join(DATA_FOLDER, 'jester', 'jester_items.csv')

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    items = pd.read_csv(items_path)
    train = train.merge(items, on='item_id')
    test = test.merge(items, on='item_id')

    average_rating = train.rating.mean()
    item_means = train.groupby('item_id').rating.agg({'mean', 'count'}).rename(
        columns={'mean': 'item_ratings_mean', 'count': 'item_ratings_count'})[
        ['item_ratings_count', 'item_ratings_mean']]

    train = train.merge(item_means, left_on='item_id', right_index=True)
    train.item_ratings_mean = (
            (train.item_ratings_mean * train.item_ratings_count - train.rating) / (train.item_ratings_count - 1))
    train.item_ratings_mean = train.item_ratings_mean.fillna(average_rating)
    train.item_ratings_count -= 1

    test = test.merge(item_means, left_on='item_id', right_index=True, how='left')
    test.item_ratings_mean = test.item_ratings_mean.fillna(average_rating)
    test.item_ratings_count = test.item_ratings_count.fillna(0)
    if subset:
        return train.drop('item_id', axis=1), test.drop('item_id', axis=1)
    return train, test
