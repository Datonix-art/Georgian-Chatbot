import json
import os

output_dir = os.path.join(os.getcwd(), 'chatbot', 'output')
combined_path = os.path.join(output_dir, 'georgian_wiki_dataset.jsonl')

with open(combined_path, 'w', encoding='utf-8') as outfile:
    batch_files = [f for f in os.listdir(output_dir) if f.startswith('georgian_wiki_batch_') and f.endswith('.jsonl')]
    batch_files.sort()
    for batch_file in batch_files:
        with open(os.path.join(output_dir, combined_path), 'r', encoding='utf-8') as infile:
            for line in infile:
                outfile.write(line)