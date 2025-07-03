""" Imports for downloading and accessing file"""
import os
import bz2
import subprocess
import sys

""" Variables """
url = 'https://dumps.wikimedia.org/kawiki/20250620/kawiki-20250620-pages-articles.xml.bz2'
file_name = url.split('/')[-1]
folder_name = os.path.join(os.getcwd(), 'chatbot', 'input')
os.makedirs(folder_name, exist_ok=True)
file_path = os.path.join(folder_name, file_name)
output_dir = os.path.join(os.getcwd(), 'chatbot', 'output')
os.makedirs(output_dir, exist_ok=True)

#1
def download_with_verification(url, file_path, max_storage=None):
    """ Download file with verification and resume capability"""
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Checking integrity...")
        try:
            if file_path.endswith('.bz2'):
                with bz2.open(file_path, 'rt', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i > 10: # read first 10 lines
                            break
                print('File integrity check passed.')
            else:
                with open(file_path, 'rt', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i > 10:
                            break
                print('File integrity check passed.')
        except Exception as e:
            print(f"Error opening the file: {e}. It may be corrupted or missing")
            print("Re-downloading file.")
            os.remove(file_path)
        
    try:
        """ Install file from specified url with wget or with curl """
        result = subprocess.run(['wget', '-c', url, '-O', file_path], capture_output=True, text=True) # capture_output enables debugging parameters for result. 
        if result.returncode != 0: 
            raise Exception('wget failed') 
    except Exception as e:
        print("Trying to download with curl instead...")
        result = subprocess.run(['curl', '-L', '-C', '-', url, '-o', file_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Download failed: {result.stderr}")
            return False

    """ Verify Download"""
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        print("File has successfully downloaded")
        print(f"File location at: {folder_name}. File size: {os.path.getsize(file_path)} bytes.")
        return True
    else:
        print("File doesnt exist. File not found or empty.")
        return False

#2
""" Validate download """    
if not download_with_verification(url, file_path):
    print('Failed to download file. Exiting')
    sys.exit(1)

#3
""" Test line integrity of downloaded file"""
print("testing line integrity")
try:
    with bz2.open(file_path, 'rt', encoding='utf-8') as f:
        line_count = 0
        for i, line in enumerate(f):
            if i < 20:
                print(f"Line {i}: {line.strip()[:100]}...")  # Show first 100 chars
            line_count += 1
            if i > 100:  # Test more lines
                break
    print(f"File integrity test passed. Read {line_count} lines successfully.")
except Exception as e:
    print(f"File integrity test failed: {e}")
    print("Please re-download the file.")
    sys.exit(1)

#4
""" Imports for cleaning wikitext"""
""" Use wikitextparser if available or else use regex-based cleaning"""
import re
try:
    import wikitextparser as wtp
    USE_WIKITEXTPARSER = True
    print('Using wikitextparser for text cleaning')
except ImportError:
    USE_WIKITEXTPARSER = False
    print('wikitextparser library is not available, using regex-based cleaning')

#5
def clean_wikitext(text):
    """ Clean wikitext markup using WikiTextParser to extract plain text"""
    if USE_WIKITEXTPARSER:
        try:
            parsed = wtp.parse(text)
            
            # Remove tables
            for table in list(parsed.tables):
                if table and table.string:
                    table.string = ''
                
            # Remove parser functions
            for pf in list(parsed.parser_functions):
                if pf and pf.string:
                    pf.string = ''
                
            # Remove comments
            for comment in list(parsed.comments):
                if comment and comment.string:
                    comment.string = ''

           # Convert wikitext to plain text
            for wikilink in list(parsed.wikilinks):
                if wikilink:
                    if wikilink.text:
                        wikilink.string = wikilink.text
                    elif wikilink.title:
                        wikilink.string = wikilink.title
                    else:
                        wikilink.string = ''
           
            # Conert external links to plain text
            for extlink in list(parsed.external_links):
                if extlink:
                    extlink.string = extlink.text if extlink.text else ''
             
            # get plain text

            plain_text = parsed.string.strip()
                
            return plain_text    
        except Exception as e:
            print(f'Error parsing text from wikitextparser: {e}')
            plain_text = regex_clean_wikitext(text)
    else:
        plain_text = regex_clean_wikitext(text)

def regex_clean_wikitext(text):
    """ Fallback regex-based cleaning for wikitext"""
    plain_text = text
   
    #* re.sub() function replaces one or many matches with a string
    
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

#6
def process_page(page_data):
    """ Process single wikipedia page"""
    try:
        title, text, page_id, ns = page_data

        # Skip non namespace articles
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

        # Skip if cleaned text size will be less then 200
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
            'len': len(clean_text)
        }
       
    except Exception as e:
        print(f"Error processing page {page_data[0] if len(page_data) > 0 else 'unknown'}: {e}")
        return None

#8
""" Imports for processing wikipedia dump file and creating ready jsonl files"""    
import multiprocessing 
import xml.etree.ElementTree as ET # Module implements simple and efficient API for parsing and creating XML data (ese ET only with trusted xml files)
from concurrent.futures import ProcessPoolExecutor
import traceback
import json

#7    
def process_wiki_dump(dump_file_path, output_dir, max_workers=None):
    """Parse wiki dump file and extract clean text"""
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    print(f'Using {max_workers} processes.')

    os.makedirs(output_dir, exist_ok=True) # Create output directory
     
    # Open the dump file and assing it to variable
    try:
        if dump_file_path.endswith('.bz2'):
            file_handle = bz2.open(dump_file_path, 'rt', encoding='utf-8')
        else:
            file_handle = open(dump_file_path, 'rt', encoding='utf-8')
    except Exception as e:
        print(f"Error opening file: {e}")
        return 0
    
    pages_processed = 0  # Counts how many pages have been read and processed
    pages_saved = 0 # Counts how many pages have been saved (Some might have been filtered/deleted )
    batch_size = 100 # Defines how many pages you want to process at once
    batch_num = 0 # Tracks current batch numbers
    
    try:
        print("Starting XML parsing...")
        # Parse XML iteratively to handle large files 
        context = ET.iterparse(file_handle, events=('start', 'end'))
        context = iter(context)
        
        # Skip to root element
        try:
            event, root = next(context)
        except StopIteration:
            print("Empty XML file")
            return 0
        
        page_batch = []
        current_page = {}
        in_page = False
        
        for event, elem in context:
            try:
                if event == 'start':
                    if elem.tag.endswith('page'):
                        current_page = {}
                        in_page = True
                elif event == 'end':
                    if elem.tag.endswith('title') and in_page:
                        current_page['title'] = elem.text or ''
                    elif elem.tag.endswith('id') and in_page and 'id' not in current_page:
                        current_page['id'] = elem.text or ''
                    elif elem.tag.endswith('ns') and in_page:
                        current_page['ns'] = elem.text or '0'
                    elif elem.tag.endswith('text') and in_page:
                        current_page['text'] = elem.text or ''
                    elif elem.tag.endswith('page'):
                        in_page = False
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
                            
                            try:
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
                                    avg_length = sum(r['len'] for r in valid_results) / len(valid_results)
                                    print(f"Batch {batch_num}: Processed {len(page_batch)} pages, saved {len(valid_results)} valid pages (avg length: {avg_length:.0f} chars)")
                                else:
                                    print(f"Batch {batch_num}: No valid pages found")
                                
                                pages_processed += len(page_batch)
                                page_batch = []
                                batch_num += 1
                                
                            except Exception as e:
                                print(f"Error processing batch {batch_num}: {e}")
                                traceback.print_exc()
                                page_batch = []
                                batch_num += 1
                    else:
                        # Clear processed elements to save memory
                        elem.clear()
                        
            except Exception as e:
                print(f"Error parsing XML element: {e}")
                continue
        
        # Process remaining pages
        if page_batch:
            print(f"Processing final batch with {len(page_batch)} pages...")
            try:
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
                print(f"Error processing final batch: {e}")
                traceback.print_exc()
    
    except Exception as e:
        print(f"Error during parsing: {e}")
        traceback.print_exc()
    finally:
        file_handle.close()
    
    print(f"Extraction complete! Processed {pages_processed} pages, saved {pages_saved} valid Georgian pages")
    
    # Create a combined file - FIXED
    combined_file = os.path.join(output_dir, 'georgian_wiki_dataset.jsonl')
    print(f"Creating combined dataset file: {combined_file}")
    
    try:
        with open(combined_file, 'w', encoding='utf-8') as combined:
            batch_files = [f for f in os.listdir(output_dir) if f.startswith('georgian_wiki_batch_') and f.endswith('.jsonl')]
            batch_files.sort()
            
            total_articles = 0
            total_length = 0
            
            for batch_file in batch_files:
                batch_path = os.path.join(output_dir, batch_file)
                try:
                    with open(batch_path, 'r', encoding='utf-8') as batch:
                        for line in batch:
                            if line.strip():  # Skip empty lines
                                combined.write(line)
                                try:
                                    data = json.loads(line)
                                    total_articles += 1
                                    total_length += data['length']
                                except json.JSONDecodeError:
                                    print(f"Warning: Invalid JSON line in {batch_file}")
                                    continue
                except Exception as e:
                    print(f"Error reading batch file {batch_file}: {e}")
                    continue
        
        avg_article_length = total_length / total_articles if total_articles > 0 else 0
        print(f"Combined dataset created with {total_articles} articles")
        print(f"Average article length: {avg_article_length:.0f} characters")
        print(f"Combined file location: {combined_file}")
        
    except Exception as e:
        print(f"Error creating combined file: {e}")
        traceback.print_exc()
    
    return pages_saved

#8
"""run file"""
if __name__ == "__main__":
    multiprocessing.freeze_support()
    cpu_count = multiprocessing.cpu_count()
    print(f'Number of available CPU cores: {cpu_count}')

    # install required packages
    print("Make sure you have installed:")
    print("pip install wikitextparser")
    print("Or the script will fall back to regex-based cleaning")

    # run extraction
    try:
        total_pages = process_wiki_dump(file_path, output_dir, max_workers=cpu_count)

        print(f"\n=== EXTRACTION SUMMARY ===")
        print(f"Total valid Georgian articles extracted: {total_pages}")
        print(f"Output directory: {output_dir}")
        print(f"Files created:")
        # List actual files created
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            batch_files = [f for f in files if f.startswith('georgian_wiki_batch_')]
            combined_file = 'georgian_wiki_dataset.jsonl'
            
            print(f"- {len(batch_files)} batch files: georgian_wiki_batch_*.jsonl")
            if combined_file in files:
                file_size = os.path.getsize(os.path.join(output_dir, combined_file))
                print(f"- Combined dataset: {combined_file} ({file_size:,} bytes)")
            else:
                print("- Combined dataset: NOT CREATED (check for errors above)")
        
        print(f"Ready for fine-tuning!")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()