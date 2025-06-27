import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer, 
    pipeline 
)
from datasets import Dataset
import json
import os
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeorgianChatBot:
    def __init__(self, model_name: str = "xml-roberta-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.qa_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def load_pretrained_model(self):
        """Load LLM"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.qa_model = AutoModelForQuestionAnswering(self.model_name)
            self.qa_model.to(self.device)
            logger.info(f"Succesfully loaded model: {self.model_name} on device: {self.device}")
            return True
        except Exception as e:
            logger.error(f"Error Loading model: {e}")
            return False
    
    def prepare_georgian_data(self, data_path: str) -> Dataset:
        """prepare georgian data for training"""

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        formatted_data = []
        for item in data:

            formatted_data.append({
                
            })

        return Dataset(formatted_data)
    
    def tokenize_data(self, dataset: Dataset) -> Dataset:
        """tokenize data for training"""
        pass

    def fine_tune(self, train_dataset: Dataset, output_dir: str, num_epochs: int = 3):
        """Fine-tune the model on georgian data"""
        pass

    def load_fine_tuned_model(self, model_path: str):
        """Load the fine-tuned Georgian model"""
        pass

    def answer_question(self, question: str, context: str):
        """Answer question based on the given context"""
        pass