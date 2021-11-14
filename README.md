# Social-network-Graph-Link-Prediction---Facebook
Taken data from facebook's recruting challenge on kaggle https://www.kaggle.com/c/FacebookRecruiting

Problem statement:
Given a directed social graph, have to predict missing links to recommend users (Link Prediction in graph).

Data Overview
Taken data from facebook's recruting challenge on kaggle https://www.kaggle.com/c/FacebookRecruiting
data contains two columns source and destination eac edge in graph - Data columns (total 2 columns):
- source_node int64
- destination_node int64

Mapping the problem into supervised learning problem:
Generated training samples of good and bad links from given directed graph and for each link got some features like no of followers, is he followed back, page rank, katz score, adar index, some svd fetures of adj matrix, some weight features etc. and trained ml model based on these features to predict link.
