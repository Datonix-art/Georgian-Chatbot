# scripts/fine_tune_georgian_qa.py
"""
Fine-tune mT5 model for Georgian Q&A
"""
import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import evaluate
import wandb
from tqdm import tqdm

@dataclass
class TrainingConfig:
    model_name: str = "google/mt5-base"
    output_dir: str = "models/georgian-qa-mt5"
    data_dir: str = "data/processed"
    
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    max_source_length: int = 512
    max_target_length: int = 256
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Hardware
    use_cuda: bool = True
    gradient_accumulation_steps: int = 2
    
    # Experiment tracking
    wandb_project: str = "georgian-qa-chatbot"
    wandb_run_name: str = "mt5-base-georgian-qa"

class GeorgianQATrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.__dict__
        )
        
        self.tokenizer = None
        self.model = None
        self.datasets = None
        
    def load_tokenizer_and_model(self):
        """Load tokenizer and model"""
        print(f"Loading tokenizer and model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
        
        # Add Georgian-specific tokens if needed
        special_tokens = {
            "additional_special_tokens": [
                "<კითხვა>", "<პასუხი>", "<კონტექსტი>", "<დასასრული>"
            ]
        }
        
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens)
        if num_added_tokens > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            print(f"Added {num_added_tokens} special tokens")
    
    def load_datasets(self):
        """Load and prepare datasets"""
        print(f"Loading datasets from {self.config.data_dir}")
        
        datasets = {}
        for split in ['train', 'validation', 'test']:
            file_path = Path(self.config.data_dir) / f'georgian_qa_{split}.jsonl'
            
            if file_path.exists():
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                
                datasets[split] = Dataset.from_list(data)
                print(f"Loaded {len(data)} examples for {split}")
            else:
                print(f"Warning: {file_path} not found")
        
        self.datasets = DatasetDict(datasets)
    
    def preprocess_function(self, examples):
        """Preprocess examples for training"""
        # Format input: "კითხვა: {question} კონტექსტი: {context}"
        inputs = []
        targets = []
        
        for i in range(len(examples['question'])):
            # Input format
            input_text = f"<კითხვა> {examples['question'][i]} <კონტექსტი> {examples['context'][i][:300]}"
            inputs.append(input_text)
            
            # Target format
            target_text = f"<პასუხი> {examples['answer'][i]} <დასასრული>"
            targets.append(target_text)
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.config.max_target_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    def prepare_datasets(self):
        """Prepare datasets for training"""
        print("Preprocessing datasets...")
        
        # Apply preprocessing
        tokenized_datasets = self.datasets.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.datasets["train"].column_names,
            desc="Tokenizing datasets"
        )
        
        return tokenized_datasets
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        
        # Replace -100 with pad token id
        labels = [[token for token in label if token != -100] for label in labels]
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Load metrics
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")
        
        # Compute ROUGE scores
        rouge_result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        # Compute BLEU score
        bleu_result = bleu.compute(
            predictions=decoded_preds,
            references=[[ref] for ref in decoded_labels]
        )
        
        return {
            "rouge1": rouge_result["rouge1"],
            "rouge2": rouge_result["rouge2"],
            "rougeL": rouge_result["rougeL"],
            "bleu": bleu_result["bleu"]
        }
    
    def train(self):
        """Train the model"""
        print("Starting training...")
        
        # Load model and tokenizer
        self.load_tokenizer_and_model()
        
        # Load and prepare datasets
        self.load_datasets()
        tokenized_datasets = self.prepare_datasets()
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            
            # Evaluation
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            
            # Model saving
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_rouge1",
            greater_is_better=True,
            
            # Misc
            predict_with_generate=True,
            generation_max_length=self.config.max_target_length,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="wandb",
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            return_tensors="pt"
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Evaluate on test set
        if "test" in tokenized_datasets:
            test_results = trainer.evaluate(tokenized_datasets["test"])
            print(f"Test results: {test_results}")
            
            # Save test results
            with open(os.path.join(self.config.output_dir, "test_results.json"), "w") as f:
                json.dump(test_results, f, indent=2)
        
        wandb.finish()
        print(f"Training completed! Model saved to {self.config.output_dir}")

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune mT5 for Georgian Q&A")
    parser.add_argument("--config-file", help="Path to config JSON file")
    parser.add_argument("--data-dir", default="data/processed", help="Data directory")
    parser.add_argument("--output-dir", default="models/georgian-qa-mt5", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    # Load config
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    
    # Create trainer and train
    trainer = GeorgianQATrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()