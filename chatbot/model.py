import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline
)
from datasets import Dataset
import json
import os
from typing import List, Dict, Tuple, Optional
import logging
import numpy as np
from torch.utils.data import DataLoader
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeorgianChatBot:
    def __init__(self, model_name: str = ""):
        """
        Initialize Georgian ChatBot
        
        Args:
            model_name: Pre-trained model which supports Georgian (name of model is: ))
        """
        self.model_name = model_name
        self.tokenizer = None
        self.qa_model = None
        self.lm_model = None  # For text generation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 512
        self.knowledge_base = []  # Store processed Georgian articles
        logger.info(f"Using device: {self.device}")
    
    def load_pretrained_model(self, task_type: str = "qa"):
        """
        Load pre-trained model
        
        Args:
            task_type: Either "qa" for question answering or "generation" for text generation
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if task_type == "qa":
                self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
                self.qa_model.to(self.device)
                logger.info(f"Successfully loaded QA model: {self.model_name} on device: {self.device}")
            elif task_type == "generation":
                self.lm_model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.lm_model.to(self.device)
                logger.info(f"Successfully loaded generation model: {self.model_name} on device: {self.device}")
            else:
                raise ValueError("task_type must be either 'qa' or 'generation'")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
   
    def prepare_georgian_data(self, data_path: str) -> Dataset:
        """
        Process Georgian Wikipedia JSONL files and create a Hugging Face dataset.
        
        Args:
            data_path: Path to the JSONL file containing Georgian Wikipedia articles
            
        Returns:
            Dataset: Hugging Face dataset with processed articles
        """
        try:
            formatted_data = []
            
            if not os.path.exists(data_path):
                logger.error(f"Data file not found: {data_path}")
                return Dataset.from_list([])
            
            logger.info(f"Loading data from: {data_path}")
            
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        if line.strip():  # Skip empty lines
                            item = json.loads(line)
                            
                            # Ensure text is not too long
                            text = item.get("text", "")
                            if len(text) > 10000:  # Limit to 10k characters
                                text = text[:10000]
                            
                            formatted_data.append({
                                "id": item.get("id", str(line_num)),
                                "title": item.get("title", ""),
                                "text": text,
                                "url": item.get("url", ""),
                                "length": len(text)
                            })
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
            
            # Store knowledge base for retrieval
            self.knowledge_base = formatted_data
            
            dataset = Dataset.from_list(formatted_data)
            logger.info(f"Successfully loaded {len(dataset)} articles")
            return dataset
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return Dataset.from_list([])
   
    def tokenize_data_for_qa(self, dataset: Dataset) -> Dataset:
        """
        Tokenize data for question answering training
        Generate question-answer pairs from the articles
        """
        def generate_qa_pairs(examples):
            """Generate question-answer pairs from article text"""
            questions = []
            contexts = []
            answers = []
            
            for i, text in enumerate(examples['text']):
                title = examples['title'][i]
                
                # Generate simple questions based on the title and content
                # This is a basic approach - you might want to use more sophisticated methods
                
                # Question about the title
                questions.append(f"რა არის {title}?")  # "What is {title}?"
                contexts.append(text)
                
                # Find the first sentence as answer
                sentences = text.split('.')
                if sentences:
                    answer_text = sentences[0].strip()
                    if answer_text:
                        answers.append({
                            'text': answer_text,
                            'answer_start': 0
                        })
                    else:
                        answers.append({
                            'text': title,
                            'answer_start': 0
                        })
                else:
                    answers.append({
                        'text': title,
                        'answer_start': 0
                    })
            
            return {
                'question': questions,
                'context': contexts,
                'answers': answers
            }
        
        # Generate QA pairs
        qa_dataset = dataset.map(generate_qa_pairs, batched=True, remove_columns=dataset.column_names)
        
        # Tokenize the QA pairs
        def tokenize_qa(examples):
            questions = [q.strip() for q in examples['question']]
            contexts = [c.strip() for c in examples['context']]
            
            # Tokenize questions and contexts
            tokenized = self.tokenizer(
                questions,
                contexts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Find answer positions in tokenized text
            start_positions = []
            end_positions = []
            
            for i, answer in enumerate(examples['answers']):
                answer_text = answer['text']
                
                # Find answer span in tokenized sequence
                # This is a simplified approach
                answer_tokens = self.tokenizer(answer_text, add_special_tokens=False)['input_ids']
                
                # For simplicity, we'll use the first few tokens as answer span
                start_positions.append(1)  # After [CLS]
                end_positions.append(min(len(answer_tokens), 10))  # Limit answer length
            
            tokenized['start_positions'] = start_positions
            tokenized['end_positions'] = end_positions
            
            return tokenized
        
        tokenized_dataset = qa_dataset.map(tokenize_qa, batched=True)
        return tokenized_dataset
    
    def tokenize_data_for_generation(self, dataset: Dataset) -> Dataset:
        """
        Tokenize data for causal language modeling (text generation)
        """
        def tokenize_function(examples):
            # Combine title and text for training
            texts = []
            for i, text in enumerate(examples['text']):
                title = examples['title'][i]
                combined_text = f"{title}\n{text}"
                texts.append(combined_text)
            
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].clone()
            
            return tokenized
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def fine_tune(self, train_dataset: Dataset, output_dir: str, task_type: str = "generation", 
                  num_epochs: int = 3, batch_size: int = 4, learning_rate: float = 5e-5):
        """
        Fine-tune the model on Georgian data
        
        Args:
            train_dataset: Tokenized training dataset
            output_dir: Directory to save the fine-tuned model
            task_type: Either "qa" or "generation"
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Prepare tokenized dataset based on task type
            if task_type == "qa":
                if self.qa_model is None:
                    logger.error("QA model not loaded. Call load_pretrained_model('qa') first.")
                    return False
                
                tokenized_dataset = self.tokenize_data_for_qa(train_dataset)
                model = self.qa_model
                data_collator = None
                
            elif task_type == "generation":
                if self.lm_model is None:
                    logger.error("Generation model not loaded. Call load_pretrained_model('generation') first.")
                    return False
                
                tokenized_dataset = self.tokenize_data_for_generation(train_dataset)
                model = self.lm_model
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False  # Causal LM, not masked LM
                )
            else:
                raise ValueError("task_type must be either 'qa' or 'generation'")
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                save_steps=500,
                save_total_limit=2,
                prediction_loss_only=True,
                learning_rate=learning_rate,
                warmup_steps=100,
                logging_dir=os.path.join(output_dir, 'logs'),
                logging_steps=50,
                report_to=None,  # Disable wandb/tensorboard
                dataloader_pin_memory=False,
                remove_unused_columns=False,
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Start training
            logger.info(f"Starting fine-tuning for {task_type} task...")
            trainer.train()
            
            # Save the model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Fine-tuning completed. Model saved to: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            return False
    
    def load_fine_tuned_model(self, model_path: str, task_type: str = "generation"):
        """
        Load the fine-tuned Georgian model
        
        Args:
            model_path: Path to the fine-tuned model
            task_type: Either "qa" or "generation"
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if task_type == "qa":
                self.qa_model = AutoModelForQuestionAnswering.from_pretrained(model_path)
                self.qa_model.to(self.device)
            elif task_type == "generation":
                self.lm_model = AutoModelForCausalLM.from_pretrained(model_path)
                self.lm_model.to(self.device)
            else:
                raise ValueError("task_type must be either 'qa' or 'generation'")
            
            logger.info(f"Successfully loaded fine-tuned model from: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            return False
    
    def find_relevant_context(self, question: str, top_k: int = 3) -> List[str]:
        """
        Find relevant contexts from knowledge base using simple similarity
        
        Args:
            question: User's question
            top_k: Number of top contexts to return
            
        Returns:
            List of relevant context strings
        """
        if not self.knowledge_base:
            return []
        
        # Simple keyword-based matching (you might want to use more sophisticated methods)
        question_words = set(question.lower().split())
        
        scored_articles = []
        for article in self.knowledge_base:
            title_words = set(article['title'].lower().split())
            text_words = set(article['text'].lower().split())
            
            # Calculate simple overlap score
            title_overlap = len(question_words & title_words)
            text_overlap = len(question_words & text_words)
            
            score = title_overlap * 2 + text_overlap  # Weight title matches more
            
            if score > 0:
                scored_articles.append((score, article))
        
        # Sort by score and return top_k
        scored_articles.sort(key=lambda x: x[0], reverse=True)
        
        contexts = []
        for score, article in scored_articles[:top_k]:
            context = f"{article['title']}\n{article['text'][:1000]}"  # Limit context length
            contexts.append(context)
        
        return contexts
    
    def answer_question(self, question: str, context: str = None, max_length: int = 200):
        """
        Answer question based on the given context or knowledge base
        
        Args:
            question: User's question
            context: Optional context. If None, will search knowledge base
            max_length: Maximum length of generated answer
            
        Returns:
            Generated answer string
        """
        try:
            if context is None:
                # Find relevant contexts from knowledge base
                contexts = self.find_relevant_context(question)
                if not contexts:
                    return "მაპატიეთ, ამ კითხვაზე პასუხი ვერ ვიპოვე ჩემს ცოდნის ბაზაში."
                context = "\n".join(contexts)
            
            if self.qa_model is not None:
                # Use QA model
                inputs = self.tokenizer(
                    question,
                    context,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.qa_model(**inputs)
                    start_scores = outputs.start_logits
                    end_scores = outputs.end_logits
                    
                    # Find the tokens with the highest scores
                    start_idx = torch.argmax(start_scores)
                    end_idx = torch.argmax(end_scores) + 1
                    
                    # Extract answer
                    answer_tokens = inputs['input_ids'][0][start_idx:end_idx]
                    answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                    
                    return answer if answer else "პასუხი ვერ ვიპოვე."
            
            elif self.lm_model is not None:
                # Use generation model
                prompt = f"კითხვა: {question}\nკონტექსტი: {context[:500]}\nპასუხი:"
                
                inputs = self.tokenizer(
                    prompt,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.lm_model.generate(
                        inputs['input_ids'],
                        max_length=inputs['input_ids'].shape[1] + max_length,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                    
                    # Decode and extract answer
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    answer = generated_text[len(prompt):].strip()
                    
                    return answer if answer else "პასუხი ვერ ვიპოვე."
            
            else:
                return "მოდელი არ არის ჩატვირთული."
        
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "შეცდომა მოხდა პასუხის გენერირებისას."
    
    def chat(self):
        """
        Start interactive chat session
        """
        print("გამარჯობა! მე ვარ ქართული ჩატბოტი. შეგიძლიათ დამისვათ კითხვები.")
        print("დაწერეთ 'exit' ან 'გასვლა' დასასრულებლად.")
        
        while True:
            try:
                question = input("\nთქვენი კითხვა: ")
                
                if question.lower() in ['exit', 'გასვლა', 'quit']:
                    print("ნახვამდის!")
                    break
                
                if not question.strip():
                    print("გთხოვთ, დაწერეთ კითხვა.")
                    continue
                
                print("ვფიქრობ...")
                answer = self.answer_question(question)
                print(f"პასუხი: {answer}")
                
            except KeyboardInterrupt:
                print("\nნახვამდის!")
                break
            except Exception as e:
                print(f"შეცდომა: {e}")


if __name__ == "__main__":
    # Example usage
    bot = GeorgianChatBot()
    
    # Load data
    data_path = os.path.join(os.getcwd(), 'chatbot', 'output', 'georgian_wiki_dataset.jsonl')
    dataset = bot.prepare_georgian_data(data_path)
    
    if len(dataset) > 0:
        print(f'Number of articles loaded: {len(dataset)}')
        print("Sample entry:", dataset[0])
        
        # Load pre-trained model
        if bot.load_pretrained_model(task_type="generation"):
            print("Model loaded successfully!")
            
            # Optional: Fine-tune the model
            output_dir = os.path.join(os.getcwd(), 'chatbot', 'fine_tuned_model')
            bot.fine_tune(dataset, output_dir, task_type="generation", num_epochs=1)
            
            # Start chat
            bot.chat()
        else:
            print("Failed to load model")
    else:
        print("No data loaded. Please check the data file path.")