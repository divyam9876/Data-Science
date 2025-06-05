#!/usr/bin/env python
import sys
import math
import json
from collections import defaultdict

# Function to calculate entropy
def calculate_entropy(values):
    total = len(values)
    counts = defaultdict(int)
    for value in values:
        counts[value] += 1
    entropy = 0.0
    for count in counts.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy

# Function to calculate information gain
def calculate_information_gain(feature_values, target_values):
    parent_entropy = calculate_entropy(target_values)

    feature_groups = defaultdict(list)
    for feature, target in zip(feature_values, target_values):
        feature_groups[feature].append(target)

    total_len = len(target_values)
    weighted_child_entropy = 0.0
    for group in feature_groups.values():
        weight = len(group) / total_len
        weighted_child_entropy += weight * calculate_entropy(group)

    return parent_entropy - weighted_child_entropy

# Class to represent Decision Tree
class DecisionTree:
    def __init__(self):
        self.tree = {}

    def build_tree(self, data, features, target_column):
        # Check if all target values are the same
        target_values = [row[target_column] for row in data]
        if len(set(target_values)) == 1:
            return target_values[0]

        # If no features left to split, return the most frequent target value
        if len(features) == 0:
            return max(set(target_values), key=target_values.count)

        # Find the best feature to split on
        best_feature = self.best_split(data, features, target_column)
        tree = {best_feature: {}}
        
        # Remove the best feature from the list of features to consider
        remaining_features = [f for f in features if f != best_feature]
        
        # Split the data based on the best feature
        feature_values = set(row[best_feature] for row in data)
        for value in feature_values:
            sub_data = [row for row in data if row[best_feature] == value]
            tree[best_feature][value] = self.build_tree(sub_data, remaining_features, target_column)
        
        return tree

    def best_split(self, data, features, target_column):
        best_feature = None
        max_info_gain = -float("inf")

        for feature in features:
            feature_values = [row[feature] for row in data]
            target_values = [row[target_column] for row in data]
            info_gain = calculate_information_gain(feature_values, target_values)
            
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = feature
        
        return best_feature

# Main function to load data and build the tree
def main(input_file):
    with open(input_file, 'r') as f:
        data = []
        header = f.readline().strip().split(',')
        target_column = 'MAX_SEV'  # Specify the target column name here
        features = [name for name in header if name != target_column]

        for line in f:
            values = line.strip().split(',')
            row = {header[i]: values[i] for i in range(len(header))}
            data.append(row)

    tree = DecisionTree()
    decision_tree = tree.build_tree(data, features, target_column)
    
    # Save the decision tree to a JSON file
    with open('dtree.json', 'w') as f:
        json.dump(decision_tree, f)

    print("Decision Tree Built and Saved Successfully to dtree.json")

if __name__ == '__main__':
    input_file = '/Users/divyamaddipatla/Desktop/Final/new/input/train_f_data.csv'  # Provide your input file path here
    main(input_file)
