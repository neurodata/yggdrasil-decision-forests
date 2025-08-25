from treeple.datasets import make_trunk_classification
import pandas as pd


X, y = make_trunk_classification(100000, seed=1)


feature_names = [f'feature_{i}' for i in range(X.shape[1])]

# Create a DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Save to CSV
df.to_csv(f'benchmarks/data/trunk_data/{X.shape[0]}x{X.shape[1]}.csv', index=False)