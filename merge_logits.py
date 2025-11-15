#merge all logits in jsonl format into single file

import os
import shutil

chunks_dir = './easydistill/chunks/'
output_file = './easydistill/logits.jsonl'  # Keep as JSONL

print("Concatenating JSONL files (fastest method)...\n")

total_entries = 0

with open(output_file, 'wb') as outf:  # Open in binary mode for speed
    for i in range(5):
        logits_file = f'{chunks_dir}logits-chunk-{i}.json'

        if not os.path.exists(logits_file):
            print(f" Chunk {i}: Not found, skipping...")
            continue

        file_size = os.path.getsize(logits_file) / (1024**3)
        print(f"Processing chunk {i} ({file_size:.2f} GB)...")

        # Copy entire file content
        with open(logits_file, 'rb') as inf:
            shutil.copyfileobj(inf, outf, length=1024*1024*16)  # 16MB buffer

        # Count entries
        with open(logits_file, 'r') as inf:
            chunk_entries = sum(1 for line in inf if line.strip())

        print(f"  Chunk {i}: {chunk_entries} entries")
        total_entries += chunk_entries

print(f"\n Concatenation complete!")
print(f"Total entries: {total_entries}")
print(f"Output: {output_file}")

output_size = os.path.getsize(output_file) / (1024**3)
print(f"File size: {output_size:.2f} GB")
