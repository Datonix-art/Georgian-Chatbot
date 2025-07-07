"""
Fine-tune mT5 model for Georgian Q&A
"""
import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    IntervalStrategy,
)
import evaluate

os.makedirs(os.path.join(os.getcwd(), 'models', 'georgian-qa-mt5'), exist_ok=True)

@dataclass
class TrainingConfig:
    model_name: str = "google/mt5-base"
    output_dir: str = os.path.join(os.getcwd(), 'models', 'georgian-qa-mt5')
    data_dir: str = os.path.join(os.getcwd(), 'data', 'processed')
    
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

class GeorgianQATrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        self.datasets = None
        
    def load_tokenizer_and_model(self):
        print(f"Loading tokenizer and model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
        
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
        inputs = []
        targets = []
        
        for i in range(len(examples['question'])):
            input_text = f"<კითხვა> {examples['question'][i]} <კონტექსტი> {examples['context'][i][:300]}"
            inputs.append(input_text)
            
            target_text = f"<პასუხი> {examples['answer'][i]} <დასასრული>"
            targets.append(target_text)
        
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
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
        print("Preprocessing datasets...")
        tokenized_datasets = self.datasets.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.datasets["train"].column_names,
            desc="Tokenizing datasets"
        )
        return tokenized_datasets
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        labels = [[token for token in label if token != -100] for label in labels]
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")
        
        rouge_result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
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
        print("Starting training...")
        
        self.load_tokenizer_and_model()
        self.load_datasets()
        tokenized_datasets = self.prepare_datasets()
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            
            save_total_limit=3,
            load_best_model_at_end=True,
            eval_strategy=IntervalStrategy.STEPS or "steps",  
            save_strategy=IntervalStrategy.STEPS or "steps",
            metric_for_best_model="eval_rouge1",
            greater_is_better=True,
            
            predict_with_generate=True,
            generation_max_length=self.config.max_target_length,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="none",
        )
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            return_tensors="pt"
        )
        
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
        
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        if "test" in tokenized_datasets:
            test_results = trainer.evaluate(tokenized_datasets["test"])
            print(f"Test results: {test_results}")
            with open(os.path.join(self.config.output_dir, "test_results.json"), "w") as f:
                json.dump(test_results, f, indent=2)
        
        print(f"Training completed! Model saved to {self.config.output_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune mT5 for Georgian Q&A")
    parser.add_argument("--config-file", help="Path to config JSON file")
    parser.add_argument("--data-dir", default="data/processed", help="Data directory")
    parser.add_argument("--output-dir", default="models/georgian-qa-mt5", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    
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
    
    trainer = GeorgianQATrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
