#* 1
#
# pip install hf_transfer -q
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"


#* 2
#
import subprocess

url = 'https://dumps.wikimedia.org/kawiki/20250620/kawiki-20250620-pages-articles.xml.bz2'
file_name = url.split('/')[-1] # divides url var into parts by removing slash and then [-1] gets last part of divided list that is actuall filename
folder_name = os.path.join(os.getcwd(), 'chatbot', 'input')

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

file_path = os.path.join(folder_name, file_name)

if not os.path.exists(file_path):
   print(f"{file_name} will be downloaded to {folder_name}")
   subprocess.run(['curl', '-L', url, '-o', file_path])
else:
    print(f"{file_name} already exists. Skipping download")

#* 3
#
import bz2


with bz2.open(file_path, 'rt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 20:
            print(line.strip())
        else:
            break

#* 4
# pip install wikiextractor
import multiprocessing
import xml.etree.ElementTree as ET
import re
from concurrent.futures import ProcessPoolExecutor
import wikitextparser as wtp 

cpu_count = multiprocessing.cpu_count()

print(f'number of available CPU cores: {cpu_count}')

output_dir = os.path.join(os.getcwd(), 'chatbot', 'output')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

subprocess.run(['wikiextractor', file_path, '--output', output_dir, '--json', '--processes', str(cpu_count)])
