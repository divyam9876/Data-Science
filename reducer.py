#!/usr/bin/env python
import sys
import math
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
    # Calculate entropy of the target values (parent entropy)
    parent_entropy = calculate_entropy(target_values)

    # Group target values based on the feature values (child groups)
    feature_groups = defaultdict(list)
    for feature, target in zip(feature_values, target_values):
        feature_groups[feature].append(target)

    # Calculate weighted child entropy
    total_len = len(target_values)
    weighted_child_entropy = 0.0
    for group in feature_groups.values():
        weight = len(group) / total_len
        weighted_child_entropy += weight * calculate_entropy(group)

    # Information gain is the difference between parent and weighted child entropy
    return parent_entropy - weighted_child_entropy

# Reducer logic
def reducer():
    # Initialize data structures to store values
    current_feature = None
    feature_values = []
    target_values = []

    # Read input line by line from stdin
    for line in sys.stdin:
        line = line.strip()
        feature, value, target = line.split('\t')

        # Collect feature values and target values
        if feature == current_feature:
            feature_values.append(value)
            target_values.append(target)
        else:
            # If we are at a new feature, process the previous one and reset
            if current_feature:
                # Calculate information gain for the previous feature
                info_gain = calculate_information_gain(feature_values, target_values)
                print(f"{current_feature}\t{info_gain}")
            
            # Reset for new feature
            current_feature = feature
            feature_values = [value]
            target_values = [target]

    # Handle last feature in the dataset
    if current_feature:
        info_gain = calculate_information_gain(feature_values, target_values)
        print(f"{current_feature}\t{info_gain}")

# Call reducer function
if __name__ == "__main__":
    reducer()
