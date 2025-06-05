#!/usr/bin/env python
import sys
import math
from collections import Counter

# Function to calculate entropy
def calculate_entropy(values):
    total = len(values)
    counts = Counter(values)
    entropy = 0.0
    for count in counts.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy

# Mapper logic
def mapper():
    # Read each line from input
    for line in sys.stdin:
        line = line.strip()
        
        # Skip the header or invalid rows
        if line.startswith('REGION') or len(line.split(',')) < 15:
            continue
        
        values = line.split(',')

        # Define feature names and the target column
        feature_names = ['REGION', 'URBANICITY', 'WEIGHT','MONTH', 'DAY_WEEK', 'HOUR', 'HARM_EV',
                         'MAN_COLL', 'RELJCT2', 'TYP_INT', 'REL_ROAD', 'LGT_COND', 'WEATHER', 'ALCOHOL']
        target = values[14]  # MAX_SEV is at index 14

        # Emit (feature_name, feature_value) as the key, and the target as the value
        for idx, name in enumerate(feature_names):
            feature_value = values[idx]
            print(f"{name}\t{feature_value}\t{target}")

# Call mapper function
if __name__ == "__main__":
    mapper()
