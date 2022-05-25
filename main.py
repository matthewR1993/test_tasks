import pandas as pd
from extract_features import DatasetStats


if __name__ == '__main__':
    dataset = pd.read_pickle("output/dataset.pkl")
    stats = DatasetStats(dataset)

    print('most_common_usage:', stats.most_common_usage(2017, 11, 11))
    print('get_conversation:', stats.get_conversation(tweet_id=4))
    print('check_conversation_group:', stats.check_conversation_group(tweet_ids=(4, 5)))
    print('num_conversations:', stats.num_conversations(turns=3))
    print('num_conversations_by_author_id:', stats.num_conversations_by_author_id(author_id='MicrosoftHelps'))
    print('conversation_stats:', len(stats.conversation_stats(2017, 10, 10)))


"""
Part C: Explanation

Please be prepared to answer the following questions:
1. At Connectly, our conversational data comes in as a stream. How would you modify your
solutions to part A and B to accommodate data streams? - Note: not all features are
needed in real-time.

2. Which caching strategies can you apply to your solutions that balances the need to
reduce redundant computer vs data freshness (is data up to date). Which features do not
change with time?

3. How would you scale your solution to a dataset with greater than 1 Billion
conversations? Would you horizontally or vertically scale the machine? How would you
calculate the operational costs?
"""
