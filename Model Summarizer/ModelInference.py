import pandas as pd
import textwrap
import json
import os
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


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Define the Model
class SummarizationModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.config = None

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
    
        
#Instantiate the model

#model_name = "allenai/led-large-16384-arxiv"

#model_summarizer = SummarizationModel(model_name)
#model = model_summarizer.model
#tokenizer=model_summarizer.tokenizer


#Generate Summary for the content using the loaded model
def generate_summary(self,content):
        max_length=300
        num_beams=4
        inputs = self.tokenizer(content, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs.input_ids.to(DEVICE), max_length=max_length, num_beams=num_beams, early_stopping=True)
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary_text

#Pase each sections and subsection to generate summaries from the model
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

#Write the final summary to the summary jsonfile
#output_file = "summary_results_allenai.json"
#summarize_pdf(pdf_data, output_file)


