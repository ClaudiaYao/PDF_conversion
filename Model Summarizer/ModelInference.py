import pandas as pd
import textwrap
import json
import os
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
transformers_logger = logging.getLogger("transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)
from transformers import LEDForConditionalGeneration, LEDTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset, load_metric
import torch
from rouge import Rouge
from rouge_score import rouge_scorer

seq_length=1024
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def Tokenize_sections(section,results):

    # List to include all the sections, content and subsections and Subsection content
    summary_results = {}
    section_results = []
    section_name = section.get("Section", "")
    content = section.get("Text")
    subsections = section.get("Subsections", [])
    ground_truth = section.get("Groundtruth")



    if content and ground_truth:
      inputs = self.tokenizer(content, return_tensors="pt", max_length=seq_length, truncation=True)
      labels = self.tokenizer(ground_truth, return_tensors="pt", max_length=seq_length, truncation=True)["input_ids"]

      # Get attention mask
      input_ids = inputs["input_ids"]
      attention_mask = inputs["attention_mask"]

        # Add to results
      results.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})


    # Process the subsections if they exist
    if "Subsections" in section:
        for subsection in section["Subsections"]:
            Tokenize_sections(subsection, results)

    return results




# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Define the Model
class SummarizationModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.config = None
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()

        if model_name =="allenai/led-large-16384-arxiv":
            self.tokenizer = LEDTokenizer.from_pretrained(model_name)
            self.model = LEDForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
            self.config = LEDForConditionalGeneration.from_pretrained(model_name).config
        elif model_name =="facebook/bart-large-cnn":
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
            self.config = BartForConditionalGeneration.from_pretrained(model_name).config
        else:
            raise ValueError("Unsupported model name")

    def get_model_name(self):
        return self.model_name
    
    def forward(self, input_ids, attention_mask, labels=None):
      input_ids = input_ids.to(self.DEVICE)
      attention_mask = attention_mask.to(self.DEVICE)
      labels = labels.to(self.DEVICE) if labels is not None else None
      outputs = self.model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                        )
      loss = self.criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
      
      return loss,outputs.logits
  
    def calculate_rouge_scores(self,generated_summary, ground_truth_summary):
      scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
      scores = scorer.score(generated_summary, ground_truth_summary)
      rouge1_f1 = scores['rouge1'].fmeasure
      rouge2_f1 = scores['rouge2'].fmeasure
      rougeL_f1 = scores['rougeL'].fmeasure
      return rouge1_f1, rouge2_f1, rougeL_f1
  
    def train_model(self,train_loader):
      total_loss = 0.0  # Initialize total loss
      for data in train_loader:
        results = []
        Tokenize_sections(data, results)
        for result in results:
          inputs = result["input_ids"]
          attention_mask = result["attention_mask"]
          labels = result["labels"]
          loss,logits = model_summarizer.forward(inputs, attention_mask, labels)

        # Backward pass
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          total_loss += loss.item()
        return total_loss
    
    def validate_model(self, val_loader):
      val_loss=0.0
      total_rouge1_f1 = 0.0
      total_rouge2_f1 = 0.0
      total_rougeL_f1 = 0.0
      num_samples = 0
      with torch.no_grad():
        for val_data in val_loader:
          val_results = []
          Tokenize_sections(val_data, val_results)
          for val_result in val_results:
            input_ids = val_result["input_ids"]
            attention_mask = val_result["attention_mask"]
            labels = val_result["labels"]
            loss,logits = model_summarizer.forward(input_ids, attention_mask, labels)
            val_loss +=loss.item()      

            # Decode the predicted summary
            predicted_token_probs = torch.softmax(logits[0], dim=-1)
            predicted_summary_ids = torch.argmax(predicted_token_probs, dim=-1).tolist()
            predicted_summary = tokenizer.decode(predicted_summary_ids, skip_special_tokens=True)
            ground_truth_summary = tokenizer.decode(labels[0], skip_special_tokens=True) 

            # Calculate ROUGE scores
            rouge1_f1, rouge2_f1, rougeL_f1 = model_summarizer.calculate_rouge_scores(predicted_summary, ground_truth_summary)

            # Accumulate ROUGE scores   
            total_rouge1_f1 += rouge1_f1
            total_rouge2_f1 += rouge2_f1
            total_rougeL_f1 += rougeL_f1
            num_samples += 1

        return val_loss,total_rouge1_f1,total_rouge2_f1,total_rougeL_f1,num_samples
    
    def log_metrics(self,epoch, train_loss, val_loss, rouge_scores):
      log_file = "metrics_log.txt"
      with open(log_file, "a") as f:        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_str = f"{timestamp}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, ROUGE: {rouge_scores}\n"
        f.write(log_str)
        



#Generate Summary for the content using the loaded model
def generate_summary(self,content):
        max_length=300
        num_beams=4
        inputs = self.tokenizer(content, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs.input_ids.to(DEVICE), max_length=max_length, num_beams=num_beams, early_stopping=True)
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary_text

#Parse each sections and subsection to generate summaries from the model
def process_section(section,results,modelsummarizer):
   
         
    
    # Process the content of the each section
    section_summary_results = {}
    content = section["Text"]
    section_name=section["Section"]
    summary_text = generate_summary(modelsummarizer,content)
    section_summary_results["Section Name"] = section_name
    section_summary_results["Generated Summary"] = summary_text
    results.append(section_summary_results)
    print("Section Name: ", section_name)
    wrapped_output = textwrap.fill(str(summary_text), width=80)
    print("Generated Summary: ", wrapped_output)
        # Process the subsections if they exist
    if "Subsections" in section:
        for subsection in section["Subsections"]:
            process_section(subsection,results,modelsummarizer)


# Summarize the section contents and subsection contents
def summarize_pdf(pdf_data, output_file,modelsummarizer):
    all_results = []
    for section in pdf_data:
        process_section(section,all_results,modelsummarizer)
    with open(output_file, "w") as json_file:
        json.dump(all_results, json_file, indent=4)



