import datetime
from typing import Dict, Tuple, List
import hashlib

import numpy as np
import pandas as pd


class DatasetStats:
    def __init__(self, dataset):
        self.dataset = dataset

    def most_common_usage(self, year: int, month: int, day: int) -> Dict:
        """
        Calculates the most common message (used more than once) on a given day.
        ■ Message text is case insensitive.
        ■ The twitter author tags (i.e.) `@sprintcare` are removed
        ■ If there is no messages used more than once, it returns an empty
        json object (`{}`)
        ■ If there are multiple message that have been repeated the same number
        of times the message with the largest MD5 hash value of the
        UTF-8 encoded bytes will be returned.
        :param year:
        :param month:
        :param day:
        :return: The most common message sent on the given day (case insensitive)
        and the number of times each author has used the message for authors who’ve
        used it 5 or more times.
        """
        date = datetime.date(year, month, day)
        df = self.dataset[self.dataset['created_at'].dt.date == date]
        if df.empty:
            return {}
        values, counts = np.unique(df['text'], return_counts=True)

        if max(counts) == 1:
            return {}

        most_frequent_texts = values[np.where(counts == np.max(counts))]
        if len(most_frequent_texts) > 1:
            # resolves the conflict by comparing hashes.
            hashes = [hashlib.md5(m.encode('utf-8')).hexdigest() for m in most_frequent_texts]
            most_frequent = most_frequent_texts[np.where(hashes == np.max(hashes))]
        else:
            most_frequent = most_frequent_texts[0]

        sr = df[df['text'] == most_frequent].groupby(['author_id'])['text'].nunique()
        authors_usage = sr[sr >= 5].to_dict()
        return {
            'most_frequent': most_frequent,
            'authors_usage': authors_usage
        }

    def get_conversation(self, tweet_id: int) -> List[Dict]:
        """
        Get Conversation For Tweet ID.
        :param tweet_id: Tweet ID contained within a conversation
        :return: A conversation in the Conversation Data Model format
        """
        conv_id = self.dataset[self.dataset['tweet_id'] == tweet_id].iloc[0]['conversation_id']
        conversation = self.dataset[self.dataset['conversation_id'] == conv_id]
        return conversation.to_dict('records')

    def check_conversation_group(self, tweet_ids: Tuple[int, int]) -> bool:
        """
        Are 2 Tweet IDs part of the same conversation.
        :param tweet_ids: A tuple of 2 Tweet IDs
        :return: A boolean True/False if they are part of the same conversation.
        """
        conv_id_1 = self.dataset[self.dataset['tweet_id'] == tweet_ids[0]]
        conv_id_2 = self.dataset[self.dataset['tweet_id'] == tweet_ids[1]]
        if conv_id_1.empty or conv_id_2.empty:
            return False

        if conv_id_1.iloc[0]['conversation_id'] == conv_id_2.iloc[0]['conversation_id']:
            return True
        return False

    def num_conversations(self, turns: int) -> int:
        """
        Number of conversations with n turns.
        :param turns: Number of turns
        :return: Number of conversations with N turns
        """
        df = self.dataset[self.dataset['turn'] == turns]
        if df.empty:
            return 0
        return len(df['conversation_id'].unique())

    def num_conversations_by_author_id(self, author_id: str) -> int:
        """
        Number of conversations for a given Author ID.
        :param author_id: Author ID
        :return: Number of conversations that include that author
        """
        df = self.dataset[self.dataset['author_id'] == author_id]
        if df.empty:
            return 0
        return len(df['conversation_id'].unique())

    def conversation_stats(self, year: int, month: int, day: int) -> Dict:
        """
        Conversation statistics for a given day.
        :param year: Year
        :param month: Month
        :param day: Day
        :return: Count of conversations Mean/Max number of turns per conversation by
        business author id (author of at least one tweet in the dataset where inbound =
        False) occurring on that day. The output is structured in the following JSON
        format: {
            <author_id>: {
                "count": <count of conversations>,
                "mean": <mean num turns>,
                "max": <max num turns>
            },
            <author_id>: {
                "count": <count of conversations>
                ...
            }
        }
        """
        date = datetime.date(year, month, day)
        df = self.dataset[self.dataset['created_at'].dt.date == date]
        df = df[df['inbound'] == True]
        return pd.concat([
            df.groupby(['author_id'])['conversation_id'].agg('count').rename('count'),
            df.groupby(['author_id'])['turn'].agg('mean').rename('mean'),
            df.groupby(['author_id'])['turn'].agg('max').rename('max')
        ], axis=1).to_dict('index')
