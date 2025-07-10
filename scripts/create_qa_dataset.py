"""
Process Georgian Wikipedia data and create Q&A dataset for fine-tuning
"""
import json
import random
import re
from pathlib import Path # for handling file paths
from typing import List, Dict, Tuple
from tqdm import tqdm # for progress bars
import nltk # for NLP tools
from transformers import AutoTokenizer # for token length checking
import os # for os path operations

# Download required NLTK data for sentence splitting
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class GeorgianQADataProcessor:
    def __init__(self, wiki_data_path: str, output_dir: str):
        self.wiki_data_path = Path(wiki_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize tokenizer for length checking
        self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-small") 
        
        # Georgian question patterns
        self.question_patterns = [
            "რა არის {}?",
            "ვინ არის {}?",
            "სად არის {}?",
            "როდის იყო {}?",
            "რატომ არის {} მნიშვნელოვანი?",
            "როგორ მუშაობს {}?",
            "რისთვის გამოიყენება {}?",
            "რომელ წელს {}?",
            "ვინ შექმნა {}?",
            "სად მდებარეობს {}?",
            "რას ნიშნავს {}?",
            "რა არის {}-ის მიზანი?",
            "როგორია {}-ის ისტორია?",
            "რა არის {}-ის უპირატესობა?",
            "როგორ გამოიყენება {}?",
        ]
        
    def load_wiki_data(self) -> List[Dict]:
        """Load processed Wikipedia data"""
        data = []
        
        if self.wiki_data_path.suffix == '.jsonl':
            with open(self.wiki_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        else:
            with open(self.wiki_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        print(f"Loaded {len(data)} articles from Wikipedia data")
        return data
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences (Georgian-specific)
        sentences = re.split(r'[.!?]+\s*', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return sentences[:5]  # Take first 5 sentences
    
    def generate_questions_from_article(self, article: Dict) -> List[Tuple[str, str]]:
        """Generate Q&A pairs from an article"""
        qa_pairs = []
        title = article['title']
        text = article['text']
        
        # Extract key sentences
        sentences = self.extract_sentences(text)
        if not sentences:
            return qa_pairs
        
        # Generate questions about the title/topic
        for pattern in self.question_patterns[:5]:  # Use first 5 patterns
            try:
                question = pattern.format(title)
                # Use first 2-3 sentences as answer
                answer = '. '.join(sentences[:2])
                
                # Check if answer is meaningful
                if len(answer) > 50 and len(answer) < 500:
                    qa_pairs.append((question, answer))
            except:
                continue
        
        # Generate questions from content
        for i, sentence in enumerate(sentences[:3]):
            # Extract potential entities/topics from sentence
            words = sentence.split()
            if len(words) > 5:
                # Simple entity extraction (can be improved)
                entities = [word for word in words if len(word) > 3 and word[0].isupper()]
                
                for entity in entities[:2]:  # Max 2 entities per sentence
                    for pattern in random.sample(self.question_patterns, 3):
                        try:
                            question = pattern.format(entity)
                            answer = sentence
                            
                            if len(answer) > 30 and len(answer) < 300:
                                qa_pairs.append((question, answer))
                        except:
                            continue
        
        return qa_pairs[:10]  # Max 10 Q&A pairs per article
    
    def create_qa_dataset(self) -> List[Dict]:
        """Create Q&A dataset from Wikipedia articles"""
        wiki_data = self.load_wiki_data()
        qa_dataset = []
        
        print("Generating Q&A pairs from articles...")
        for article in tqdm(wiki_data):
            qa_pairs = self.generate_questions_from_article(article)
            
            for question, answer in qa_pairs:
                qa_dataset.append({
                    'id': f"{article['id']}_{len(qa_dataset)}",
                    'question': question,
                    'answer': answer,
                    'context': article['text'][:500],  # First 500 chars as context
                    'article_title': article['title'],
                    'article_url': article.get('url', ''),
                    'input_text': f"კითხვა: {question}\nკონტექსტი: {article['text'][:300]}...",
                    'target_text': answer
                })
        
        print(f"Generated {len(qa_dataset)} Q&A pairs")
        return qa_dataset
    
    def save_datasets(self, qa_dataset: List[Dict]):
        """Save datasets in different formats"""
        # Shuffle and split
        random.shuffle(qa_dataset)
        
        total_size = len(qa_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        train_data = qa_dataset[:train_size]
        val_data = qa_dataset[train_size:train_size + val_size]
        test_data = qa_dataset[train_size + val_size:]
        
        # Save as JSONL
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        for split_name, data in splits.items():
            # JSONL format
            jsonl_path = self.output_dir / f'georgian_qa_{split_name}.jsonl'
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"Saved {len(data)} {split_name} examples to {jsonl_path}")
        
        # Save summary
        summary = {
            'total_qa_pairs': total_size,
            'train_size': len(train_data),
            'validation_size': len(val_data),
            'test_size': len(test_data),
            'avg_question_length': sum(len(item['question']) for item in qa_dataset) / total_size,
            'avg_answer_length': sum(len(item['answer']) for item in qa_dataset) / total_size,
        }
        
        with open(self.output_dir / 'dataset_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"Dataset summary saved to {self.output_dir / 'dataset_summary.json'}")
        return splits

def main():
    """Main function to create Q&A dataset"""
    wiki_data_path = os.path.join(os.getcwd(), 'data', 'raw', 'output', 'georgian_wiki_dataset.jsonl')
    output_dir = os.path.join(os.getcwd(), 'data', 'processed')
    processor = GeorgianQADataProcessor(wiki_data_path, output_dir)
    qa_dataset = processor.create_qa_dataset()
    
    if qa_dataset:
        processor.save_datasets(qa_dataset)
        print("Dataset creation completed successfully!")
    else:
        print("No Q&A pairs generated. Check your Wikipedia data.")

if __name__ == "__main__":
    main()