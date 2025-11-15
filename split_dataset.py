import json

# Load the full 100k dataset
input_file = './distilqwen-100k.json'
output_dir = './easydistill/chunks/'

# Create chunks directory
!mkdir -p {output_dir}

# Load data
print("Loading dataset...")
with open(input_file, 'r') as f:
    full_data = json.load(f)

print(f"Total examples: {len(full_data)}")

# Split into 5 chunks of 20k each
chunk_size = 20000
num_chunks = 5

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size

    chunk_data = full_data[start_idx:end_idx]
    chunk_file = f'{output_dir}distilqwen-chunk-{i}.json'

    with open(chunk_file, 'w') as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)

    print(f" Created chunk {i}: {len(chunk_data)} examples -> {chunk_file}")

print("\n All chunks created successfully!")
