import pandas as pd
import textwrap
import json
import os
from datetime import datetime
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
transformers_logger = logging.getLogger("transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.disable(logging.INFO) # disable INFO and DEBUG logging everywhere
logging.disable(logging.WARNING) 
from transformers import LEDForConditionalGeneration, LEDTokenizer
from datasets import load_dataset, load_metric
import torch
from rouge import Rouge
from rouge_score import rouge_scorer

#Define the sequence length for the model.
seq_length=1024

def load_data(file_path):
    with open(file_path, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
#Define the Model
class SummarizationModel(nn.Module):  

    def __init__(self, model_name):
      super().__init__()
      self.model_name = model_name
      self.tokenizer = LEDTokenizer.from_pretrained(model_name)
      self.model = LEDForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
      self.config= LEDForConditionalGeneration.from_pretrained(model_name).config
      self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=0.01)
      #self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True)
      self.scheduler = CosineAnnealingLR(self.optimizer, no_epochs=10)
      self.criterion = torch.nn.CrossEntropyLoss()        


    #Forward Method
    def forward(self, input_ids, attention_mask, labels):
      input_ids = input_ids.to(DEVICE)
      attention_mask = attention_mask.to(DEVICE)
      labels = labels.to(DEVICE) 
      outputs = self.model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                        )
      logits = outputs.logits
      logits_flat = logits.view(-1, logits.size(-1))
      labels_flat = labels.view(-1)
      loss = self.criterion(logits_flat, labels_flat)
          
      return loss,logits
  
    #Function to calculate ROUGE scores for generated summary and ground truth
    def calculate_rouge_scores(self,generated_summary, ground_truth_summary):
        
      scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
      scores = scorer.score(generated_summary, ground_truth_summary)
      rouge1_f1 = scores['rouge1'].fmeasure * 100
      rouge2_f1 = scores['rouge2'].fmeasure * 100
      rougeL_f1 = scores['rougeL'].fmeasure * 100
      return rouge1_f1, rouge2_f1, rougeL_f1
    
      # Function to tokenize sections before training the model
    def  pre_process_data(self,section,results):
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
                  model_summarizer.pre_process_data(subsection, results)
        return results  
  
    
    # Function to train model
    def train_model(self,train_loader):
      self.model.train()
      total_loss = 0.0  # Initialize total loss
      for data in train_loader:
        results = []
        model_summarizer.pre_process_data(data, results)
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
    
    #Function to evaluate model using val set
    def validate_model(self, val_loader):
      self.model.eval()
      val_loss=0.0
      total_rouge1_f1 = 0.0
      total_rouge2_f1 = 0.0
      total_rougeL_f1 = 0.0
      num_samples = 0
      with torch.no_grad():
        for val_data in val_loader:
          val_results = []
          model_summarizer.pre_process_data(val_data, val_results)
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
    
    #Function to test the model using test dataset
    def test_model(self,section, model):
      self.model.eval() 
      section_summary_results = {}
      content = section["Text"]
      section_name = section["Section"]
      ground_truth_summary = section.get("Groundtruth")[0]
      if content and ground_truth_summary:
        # Tokenize the content

        inputs = self.tokenizer(content, return_tensors="pt", max_length=seq_length, truncation=True)
        labels = self.tokenizer(ground_truth_summary, return_tensors="pt", max_length=seq_length, truncation=True)["input_ids"]

        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1
        labels = labels.to(DEVICE)
        with torch.no_grad():
          outputs = model.generate(input_ids, 
                                   attention_mask=attention_mask, 
                                   global_attention_mask=global_attention_mask, 
                                   max_length=seq_length, 
                                   num_beams=4,
                                   no_repeat_ngram_size=3,
                                   early_stopping=True, 
                                   num_return_sequences=1
                                   )              

          # Decode the generated summary using the tokenizer
          summary_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
          ground_truth_summary = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Calculate ROUGE scores
          rouge1_f1, rouge2_f1, rougeL_f1 = model_summarizer.calculate_rouge_scores(summary_text, ground_truth_summary)

        section_summary_results["Section Name"] = section_name
        section_summary_results["Generated Summary"] = summary_text
        section_summary_results["ROUGE-1 F1"] = rouge1_f1
        section_summary_results["ROUGE-2 F1"] = rouge2_f1
        section_summary_results["ROUGE-L F1"] = rougeL_f1
        
        print("Section Name: ", section_name)
        wrapped_output = textwrap.fill(str(summary_text), width=80)
        print("Generated Summary: ", wrapped_output)

        if "Subsections" in section:
           for subsection in section["Subsections"]:
            model_summarizer.test_model(subsection,model)

      return section_summary_results

    
    #Function to log the experiment results
    def log_metrics(self,epoch, train_loss, val_loss, rouge_scores):
      log_file = "logs/metrics_log.txt"
      with open(log_file, "a+") as f:        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_str = f"{timestamp}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, ROUGE: {rouge_scores}\n"
        f.write(log_str)
        


#Model Inference
#Generate Summary for the content using the loaded model
    def generate_summary(self,content):
        max_length=300
        num_beams=4
        inputs = self.tokenizer(content, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs.input_ids.to(DEVICE), max_length=max_length, num_beams=num_beams, early_stopping=True)
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary_text

#Parse each sections and subsection to generate summaries from the model
    def process_section(self,section,results,modelsummarizer): 
        # Process the content of the each section
        section_summary_results = {}
        content = section["Text"]
        section_name=section["Section"]
        summary_text = model_summarizer.generate_summary(content)
        section_summary_results["Section Name"] = section_name
        section_summary_results["Generated Summary"] = summary_text
        results.append(section_summary_results)
        print("Section Name: ", section_name)
        wrapped_output = textwrap.fill(str(summary_text), width=80)
        print("Generated Summary: ", wrapped_output)
        # Process the subsections if they exist
        if "Subsections" in section:
            for subsection in section["Subsections"]:
                model_summarizer.process_section(subsection,results,modelsummarizer)
            
             
    # Summarize the section contents and subsection contents
    def summarize_pdf(self, pdf_data, output_file,modelsummarizer):
        all_results = []
        for section in pdf_data:
            model_summarizer.process_section(section,all_results,modelsummarizer)
        
        with open(output_file, "w+") as json_file:
            json.dump(all_results, json_file, indent=4)    
        
    
model_name = "allenai/led-large-16384-arxiv"
model_summarizer = SummarizationModel(model_name)
model = model_summarizer.model
tokenizer = model_summarizer.tokenizer    

