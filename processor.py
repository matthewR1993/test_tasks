import re
import datetime
from datetime import timezone
import os
import urllib.request

import numpy as np
import pandas as pd


def remove_usernames(msg: str) -> str:
    return re.sub('@[^\s]+', '', msg).strip().lower()


def make_folders(folders):
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)


def process(data: pd.DataFrame) -> pd.DataFrame:
    data['text'] = data['text'].apply(remove_usernames)
    data['created_at'] = data['created_at'].apply(lambda x: datetime.datetime.strptime(x, '%a %b %d %H:%M:%S +%f %Y'))
    data['created_at'] = data['created_at'].dt.tz_localize(timezone.utc)

    conversation_id = 0
    start_ids = data[data['in_response_to_tweet_id'].isnull()]['tweet_id'].values
    messages = []
    for start_id in start_ids:
        sr = data[data['tweet_id'] == start_id].iloc[0]
        messages.append({
            'text': sr['text'],
            'created_at': sr['created_at'],
            'author_id': sr['author_id'],
            'tweet_id': sr['tweet_id'],
            'conversation_id': conversation_id,
            'turn': 0
        })
        ids = [int(x) for x in data[data['tweet_id'] == start_id].iloc[0]['response_tweet_id'].split(',')]

        turn = 1
        while True:
            df_ = data[data['tweet_id'].isin(ids)]
            for index, row in df_.iterrows():
                messages.append({
                    'text': row['text'],
                    'created_at': row['created_at'],
                    'author_id': row['author_id'],
                    'tweet_id': row['tweet_id'],
                    'conversation_id': conversation_id,
                    'turn': turn
                })
            turn += 1

            sr = df_[~df_['response_tweet_id'].isnull()]['response_tweet_id']
            if not sr.isnull().values.any() and len(sr) > 0:
                ids = [int(x) for x in sr.values[0].split(',')]
            else:
                break
        conversation_id += 1

    df_messages = pd.DataFrame(messages)
    data['conversation_id'] = None
    data['turn'] = None
    data.loc[data['tweet_id'].isin(df_messages['tweet_id']), 'conversation_id'] = df_messages['conversation_id']
    data.loc[data['tweet_id'].isin(df_messages['tweet_id']), 'turn'] = df_messages['turn']
    mx = data['conversation_id'].max()
    data.loc[data['conversation_id'].isnull(), 'conversation_id'] = np.arange(mx + 1, mx + len(data[data['conversation_id'].isnull()]) + 1, 1, dtype=float)
    data.loc[data['turn'].isnull(), 'turn'] = 0
    data.drop(['in_response_to_tweet_id'], axis=1, inplace=True)
    data.drop(['response_tweet_id'], axis=1, inplace=True)
    return data


if __name__ == '__main__':
    make_folders(['data', 'output'])
    original_file = 'data/twcs.csv'
    if not os.path.exists(original_file):
        url = 'https://cdn.connectly.ai/datasets/kaggle_twitter_customer_support/twcs.csv'
        urllib.request.urlretrieve(url, original_file)
    data = pd.read_csv(original_file)
    # data = data.head(500000)
    data = process(data)
    data.to_pickle("output/dataset.pkl")
    print('Size in bytes:', os.stat(r'output/dataset.pkl').st_size)
