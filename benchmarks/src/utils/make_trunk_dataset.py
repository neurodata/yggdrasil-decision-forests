import argparse
from treeple.datasets import make_trunk_classification
import pandas as pd

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate trunk classification dataset')
parser.add_argument('--rows', type=int, required=True, help='Number of rows to generate')
parser.add_argument('--cols', type=int, default=4096, help='Number of cols to generate')

# Parse arguments
args = parser.parse_args()

X, y = make_trunk_classification(args.rows, n_dim=args.cols, seed=1)

feature_names = [f'feature_{i}' for i in range(X.shape[1])]

# Create a DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Save to CSV
df.to_csv(f'benchmarks/data/trunk_data/{X.shape[0]}x{X.shape[1]}.csv', index=False)