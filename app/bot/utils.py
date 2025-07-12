"""
Utility functions for Georgian Q&A chatbot
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict
import json
import re
import os
import logging
from datetime import datetime
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeorgianQABot:
    def __init__(self, model_path: str, max_length: int = 256):
        """
        Initialize the Georgian Q&A bot
        
        Args:
            model_path: Path to the fine-tuned model
            max_length: Maximum length for generated responses
        """
        self.model_path = model_path
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}\{traceback.print_exc()}")
            raise

        self.context_data = None
        self.load_context_data()

    def load_context_data(self):
        """Load context data from processed wikipedia"""
        context_file = os.path.join(os.getcwd(), 'data', 'raw', 'output', 'georgian_wiki_dataset.jsonl')
        
        if os.path.exists(context_file):
            self.context_data = []
            try:
                with open(context_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                self.context_data.append(data)
                            except json.JSONDecodeError:
                                continue
                
                logger.info(f"Loaded {len(self.context_data)} context articles")
            except Exception as e:
                logger.error(f"Error loading context data: {e}")
                self.context_data = []
        else:
            logger.warning("No context data found")
            self.context_data = []

    def find_relevant_context(self, question: str, top_k: int = 3) -> str:
        """Find relevant context for the question"""
        if not self.context_data:
            return ""
        
        question_words = set(question.lower().split())
        scored_articles = []
        
        for article in self.context_data:
            title_lower = article['title'].lower()
            text_lower = article['text'][:500].lower()  # First 500 chars
            
            # Simple scoring based on word overlap
            title_words = set(title_lower.split())
            text_words = set(text_lower.split())
            
            title_overlap = len(question_words & title_words)
            text_overlap = len(question_words & text_words)
            
            score = title_overlap * 3 + text_overlap  # Title matches are more important

            if score > 0:
                scored_articles.append((score, article))

        scored_articles.sort(reverse=True, key=lambda x: x[0])
    
        # Combine context from top articles
        context_parts = []
        for score, article in scored_articles[:top_k]:
            context_parts.append(article['text'][:200])  # First 200 chars
        
        return " ".join(context_parts)
    
    def preprocess_question(self, question: str) -> str:
        """Preprocess the question"""
        # Clean the question
        question = re.sub(r'\s+', ' ', question).strip()
        
        # Add question mark if missing
        if not question.endswith('?'):
            question += '?'
        
        return question
    
    def generate_answer(self, question: str, context: str = "") -> str:
        """
        Generate answer for the given question
        
        Args:
            question: The question to answer
            context: Optional context to help answer the question
            
        Returns:
            Generated answer
        """
        try:
            # Preprocess question
            question = self.preprocess_question(question)
            
            # Find relevant context if not provided
            if not context:
                context = self.find_relevant_context(question)
            
            # Format input
            input_text = f"<კითხვა> {question}"
            if context:
                input_text += f" <კონტექსტი> {context[:300]}"  # Limit context length
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length,
                    min_length=10,
                    num_beams=3,
                    early_stopping=True,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the answer
            answer = self.postprocess_answer(generated_text)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            traceback.print_exc()
            return "ვერ მოვძებნე შესაბამისი პასუხი."
    
    def postprocess_answer(self, generated_text: str) -> str:
        """Clean up the generated answer"""
        # Remove special tokens if any remain
        answer = generated_text.replace("<პასუხი>", "").replace("<დასასრული>", "")
        
        # Clean up whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Remove repetitive phrases
        sentences = answer.split('.')
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        answer = '. '.join(unique_sentences)
        
        # Ensure proper ending
        if answer and not answer.endswith('.'):
            answer += '.'
        
        return answer if answer else "ვერ მოვძებნე შესაბამისი პასუხი."
    
    def get_conversation_response(self, question: str, context: str = "") -> Dict:
        """
        Get a complete response with metadata
        
        Args:
            question: User's question
            context: Optional context
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            answer = self.generate_answer(question, context)
            
            return {
                "answer": answer,
                "question": question,
                "context_used": bool(context or self.find_relevant_context(question)),
                "success": True,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in conversation response: {e}")
            return {
                "answer": "ვერ მოვძებნე შესაბამისი პასუხი. გთხოვთ, სცადოთ სხვა კითხვა.",
                "question": question,
                "context_used": False,
                "success": False,
                "error": str(e)
            }

# Utility functions for the Flask app
def create_qa_bot(model_path: str = os.path.join(os.getcwd(), 'models', 'georgian-qa-mt5')) -> GeorgianQABot:
    """Create and return a Georgian Q&A bot instance"""
    try:
        return GeorgianQABot(model_path)
    except Exception as e:
        logger.error(f"Error creating QA bot: {e}")
        raise

def format_response_for_web(response: Dict) -> Dict:
    """Format response for web display"""
    return {
        "answer": response["answer"],
        "success": response["success"],
        "error": response.get("error"),
        "timestamp": datetime.now().isoformat(),
        "context_used": response.get("context_used", False)
    }