#!/usr/bin/env python
import json
import csv
from collections import defaultdict
from sklearn.metrics import classification_report

# Function to load the decision tree from a JSON file
def load_tree(filename):
    with open(filename, 'r') as f:
        tree = json.load(f)
    return tree

# Function to predict based on the decision tree
def predict(tree, row):
    if isinstance(tree, dict):
        feature = list(tree.keys())[0]
        value = row[feature]
        if value in tree[feature]:
            return predict(tree[feature][value], row)
        else:
            return None  # Handle missing or unexpected values
    else:
        return tree

# Function to handle missing predictions (fallback to the most frequent label)
def handle_missing_prediction(y_pred, y_true):
    if None in y_pred:
        # Fallback: Assign the most frequent label in the true labels to None predictions
        most_frequent_label = max(set(y_true), key=y_true.count)
        y_pred = [x if x is not None else most_frequent_label for x in y_pred]
    return y_pred

# Main function to load test data, make predictions, and evaluate accuracy
def main(test_file, tree_file):
    # Load the trained decision tree
    tree = load_tree(tree_file)

    # Load test data
    with open(test_file, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # Prepare data for predictions
    X_test = [row for row in data]
    y_true = [row['MAX_SEV'] for row in data]  # Assuming 'MAX_SEV' is the true label
    y_pred = []

    for row in X_test:
        y_pred.append(predict(tree, row))

    # Handle missing predictions by replacing None with the most frequent label
    y_pred = handle_missing_prediction(y_pred, y_true)

    # Calculate and print classification report (Accuracy, Precision, Recall, F1-score)
    print(classification_report(y_true, y_pred))

if __name__ == '__main__':
    test_file = '/Users/divyamaddipatla/Desktop/Final/new/input/test_f_data.csv'  # Provide your test file path here
    tree_file = 'dtree.json'  # The decision tree saved in training script
    main(test_file, tree_file)
