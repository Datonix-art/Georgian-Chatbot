#* 1
#
# pip install hf_transfer -q
import os

"""Environmental variable that fastens downloading models or datasets from hugging face. (Not used)"""
# os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1" 


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
# debugging
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
import xml.etree.ElementTree as ET # or import defusedxml.ElementTree as ET if untrusted XML file (!pip install defusedxml)
import re
from concurrent.futures import ProcessPoolExecutor
import wikitextparser as wtp 
import json


def clean_wikitext(text):
    """Clean wikitext markup using wikitextparser to extract plain text"""
    try:
        # Parse with wikitextparser
        parsed = wtp.parse(text)
        
        # Remove templates
        for template in parsed.templates:
            template.string = ''
        
        # Remove tables
        for table in parsed.tables:
            table.string = ''
            
        # Remove parser functions
        for pf in parsed.parser_functions:
            pf.string = ''
            
        # Remove comments
        for comment in parsed.comments:
            comment.string = ''
            
        # Convert wikilinks to plain text
        for wikilink in parsed.wikilinks:
            if wikilink.text:
                wikilink.string = wikilink.text
            elif wikilink.title:
                wikilink.string = wikilink.title
            else:
                wikilink.string = ''
        
        # Convert external links to plain text
        for extlink in parsed.external_links:
            if extlink.text:
                extlink.string = extlink.text
            else:
                extlink.string = ''
        
        # Get the plain text
        plain_text = parsed.plain()
        
    except Exception as e:
        print(f"Error parsing wikitext: {e}")
        # Fallback to regex-based cleaning
        plain_text = text
        
        # Remove templates
        plain_text = re.sub(r'\{\{[^}]*\}\}', '', plain_text)
        # Convert internal links [[link|text]] -> text or [[link]] -> link
        plain_text = re.sub(r'\[\[([^|\]]*\|)?([^\]]*)\]\]', r'\2', plain_text)
        # Convert external links [http://example.com text] -> text
        plain_text = re.sub(r'\[http[^\s]* ([^\]]*)\]', r'\1', plain_text)
        # Remove external links without text [http://example.com]
        plain_text = re.sub(r'\[http[^\s]*\]', '', plain_text)
    
    # Additional cleaning for Georgian text
    # Remove bold '''text''' -> text
    plain_text = re.sub(r"'''([^']*)'''", r'\1', plain_text)
    # Remove italic ''text'' -> text
    plain_text = re.sub(r"''([^']*)''", r'\1', plain_text)
    # Remove references <ref>...</ref>
    plain_text = re.sub(r'<ref[^>]*>.*?</ref>', '', plain_text, flags=re.DOTALL)
    plain_text = re.sub(r'<ref[^>]*/?>', '', plain_text)
    # Remove HTML tags
    plain_text = re.sub(r'<[^>]+>', '', plain_text)
    # Remove file/image references
    plain_text = re.sub(r'\[\[File:.*?\]\]', '', plain_text, flags=re.DOTALL)
    plain_text = re.sub(r'\[\[Image:.*?\]\]', '', plain_text, flags=re.DOTALL)
    # Remove category links
    plain_text = re.sub(r'\[\[Category:.*?\]\]', '', plain_text, flags=re.IGNORECASE)
    plain_text = re.sub(r'\[\[კატეგორია:.*?\]\]', '', plain_text)
    # Clean up whitespace
    plain_text = re.sub(r'\n+', '\n', plain_text)
    plain_text = re.sub(r'^\s*$', '', plain_text, flags=re.MULTILINE)
    plain_text = re.sub(r'\s+', ' ', plain_text)
    
    return plain_text.strip()

def process_page(page_data):
    """Process a single Wikipedia page"""
    title, text, page_id, ns = page_data
    
    # Skip non-main namespace articles (ns != 0)
    if ns != '0':
        return None
    
    # Skip redirect pages
    if text.strip().lower().startswith('#redirect') or text.strip().lower().startswith('#გადამისამართება'):
        return None
    
    # Skip disambiguation pages
    if 'disambiguation' in title.lower() or 'მრავალმნიშვნელოვანი' in title.lower():
        return None
    
    # Clean the text
    clean_text = clean_wikitext(text)
    
    # Skip if too short (less than 200 characters for Georgian)
    if len(clean_text) < 200:
        return None
    
    # Skip if mostly non-Georgian characters (basic check)
    georgian_chars = len(re.findall(r'[ა-ჿ]', clean_text))
    total_chars = len(re.sub(r'\s', '', clean_text))  # Count non-whitespace chars
    
    if total_chars > 0 and georgian_chars < total_chars * 0.3:  # At least 30% Georgian characters
        return None
    
    return {
        'id': page_id,
        'title': title,
        'text': clean_text,
        'url': f'https://ka.wikipedia.org/wiki/{title.replace(" ", "_")}',
        'length': len(clean_text)
    }

def parse_wikipedia_dump(dump_file_path, output_dir, max_workers=None):
    """Parse Wikipedia dump file and extract clean text"""
    
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    print(f"Using {max_workers} processes")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the dump file
    if dump_file_path.endswith('.bz2'):
        file_handle = bz2.open(dump_file_path, 'rt', encoding='utf-8')
    else:
        file_handle = open(dump_file_path, 'r', encoding='utf-8')
    
    pages_processed = 0
    pages_saved = 0
    batch_size = 500  # Smaller batch size for wikitextparser
    batch_num = 0
    
    try:
        print("Starting XML parsing...")
        # Parse XML iteratively to handle large files 
        context = ET.iterparse(file_handle, events=('start', 'end')) #Efficient memory usage: reads the XML element-by-element (not whole file).
        context = iter(context)
        event, root = next(context)
        
        page_batch = []
        current_page = {}
        
        for event, elem in context:
            if event == 'start':
                if elem.tag.endswith('page'):
                    current_page = {}
            elif event == 'end':
                if elem.tag.endswith('title'):
                    current_page['title'] = elem.text or ''
                elif elem.tag.endswith('id') and 'id' not in current_page:  # Only take the first id (page id, not revision id)
                    current_page['id'] = elem.text or ''
                elif elem.tag.endswith('ns'):
                    current_page['ns'] = elem.text or '0'
                elif elem.tag.endswith('text'):
                    current_page['text'] = elem.text or ''
                elif elem.tag.endswith('page'):
                    # Page complete, add to batch if it has all required fields
                    if all(key in current_page for key in ['title', 'text', 'id', 'ns']):
                        page_batch.append((
                            current_page['title'],
                            current_page['text'],
                            current_page['id'],
                            current_page['ns']
                        ))
                    
                    # Clear the element to save memory
                    elem.clear()
                    if elem != root:
                        root.clear()
                    
                    # Process batch when it reaches batch_size
                    if len(page_batch) >= batch_size:
                        print(f"Processing batch {batch_num} with {len(page_batch)} pages...")
                        
                        with ProcessPoolExecutor(max_workers=max_workers) as executor:
                            results = list(executor.map(process_page, page_batch))
                        
                        # Save results
                        valid_results = [r for r in results if r is not None]
                        if valid_results:
                            output_file = os.path.join(output_dir, f'georgian_wiki_batch_{batch_num:04d}.jsonl')
                            with open(output_file, 'w', encoding='utf-8') as f:
                                for result in valid_results:
                                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                            
                            pages_saved += len(valid_results)
                            avg_length = sum(r['length'] for r in valid_results) / len(valid_results)
                            print(f"Batch {batch_num}: Processed {len(page_batch)} pages, saved {len(valid_results)} valid pages (avg length: {avg_length:.0f} chars)")
                        else:
                            print(f"Batch {batch_num}: No valid pages found")
                        
                        pages_processed += len(page_batch)
                        page_batch = []
                        batch_num += 1
                else:
                    # Clear processed elements to save memory
                    elem.clear()
        
        # Process remaining pages
        if page_batch:
            print(f"Processing final batch with {len(page_batch)} pages...")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_page, page_batch))
            
            valid_results = [r for r in results if r is not None]
            if valid_results:
                output_file = os.path.join(output_dir, f'georgian_wiki_batch_{batch_num:04d}.jsonl')
                with open(output_file, 'w', encoding='utf-8') as f:
                    for result in valid_results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                pages_saved += len(valid_results)
                avg_length = sum(r['length'] for r in valid_results) / len(valid_results)
                print(f"Final batch: Processed {len(page_batch)} pages, saved {len(valid_results)} valid pages (avg length: {avg_length:.0f} chars)")
    
    except Exception as e:
        print(f"Error during parsing: {e}")
        raise
    finally:
        file_handle.close()
    
    print(f"Extraction complete! Processed {pages_processed} pages, saved {pages_saved} valid Georgian pages")
    
    # Create a combined file
    combined_file = os.path.join(output_dir, 'georgian_wikipedia_dataset.jsonl')
    print(f"Creating combined dataset file: {combined_file}")
    
    with open(combined_file, 'w', encoding='utf-8') as combined:
        batch_files = [f for f in os.listdir(output_dir) if f.startswith('georgian_wiki_batch_') and f.endswith('.jsonl')]
        batch_files.sort()
        
        total_articles = 0
        total_length = 0
        
        for batch_file in batch_files:
            batch_path = os.path.join(output_dir, batch_file)
            with open(batch_path, 'r', encoding='utf-8') as batch:
                for line in batch:
                    combined.write(line)
                    data = json.loads(line)
                    total_articles += 1
                    total_length += data['length']
    
    avg_article_length = total_length / total_articles if total_articles > 0 else 0
    print(f"Combined dataset created with {total_articles} articles")
    print(f"Average article length: {avg_article_length:.0f} characters")
    
    return pages_saved

# Get CPU count and setup output directory
cpu_count = multiprocessing.cpu_count()
print(f'Number of available CPU cores: {cpu_count}')
output_dir = os.path.join(os.getcwd(), 'chatbot', 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Run the extraction
print("Starting Wikipedia extraction with wikitextparser...")
print("Make sure you have installed: pip install wikitextparser")

try:
    total_pages = parse_wikipedia_dump(file_path, output_dir, max_workers=cpu_count)
    
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"Total valid Georgian articles extracted: {total_pages}")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"- Individual batch files: georgian_wiki_batch_*.jsonl")
    print(f"- Combined dataset: georgian_wikipedia_dataset.jsonl")
    print(f"Ready for fine-tuning!")
except Exception as e:
    print(f"Error: {e}")
