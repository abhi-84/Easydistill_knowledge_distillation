import sys
import pandas as pd

# Read the parquet file
df = pd.read_parquet(sys.argv[1])

# Create output filename by replacing .parquet extension with .json
output_file = sys.argv[1].rsplit('.', 1)[0] + '.json'

# Save to JSON file
df.to_json(output_file, orient='records', indent=2)

print(f"Successfully converted {sys.argv[1]} to {output_file}")
