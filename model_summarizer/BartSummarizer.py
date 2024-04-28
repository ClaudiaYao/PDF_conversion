import json
import os
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer, LEDForConditionalGeneration, LEDTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Dataset
from rouge_score import rouge_scorer
import torch
import textwrap

MAX_LENGTH = 244


def get_model(model_name):
	model = BartForConditionalGeneration.from_pretrained(model_name)
	tokenizer = BartTokenizer.from_pretrained(model_name)
	return model, tokenizer

def load_data(file_path, split):
	data = load_dataset('json', data_files=file_path, split=split)
	return data

def load_data_from_json(file_path):
	json_data = load_json(file_path)

	def flatten_subsections(sections):
		results = []
		for section in sections:
			results.append(section)
			results += flatten_subsections(section["Subsections"])
		return results
	
	flattened_json_data = flatten_subsections(json_data)

	return Dataset.from_list(flattened_json_data)
				

def load_json(file_path):
    with open(file_path, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def get_data_massager(tokenizer):
	def preprocess_function(sections):
		inputs = tokenizer(sections["Text"], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
		labels = tokenizer(sections["Groundtruth"], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)

		return {
			"input_ids": inputs.input_ids.to('cuda'),
			"attention_mask": inputs.attention_mask.to('cuda'),
			"labels": labels.input_ids.to('cuda'),
		}
	return preprocess_function

def get_training_args(**kwargs):
	return Seq2SeqTrainingArguments(
		output_dir='./Checkpoints',
		per_device_train_batch_size=8,
		save_strategy='epoch',
		evaluation_strategy="epoch",
		adam_beta1=0.9,
		adam_beta2=0.999,
		adam_epsilon=1e-8,
		**kwargs
	)

def train_model(model_name, **kwargs):
	torch.cuda.empty_cache()
	model, tokenizer = get_model(model_name)
	training_args = get_training_args(**kwargs)

	massage_data = get_data_massager(tokenizer)

	train_data = load_data_from_json(
		os.path.join('..', 'data', 'dataset', 'dataset_ground_truth.json'),
	)
	train_data = train_data.map(massage_data, batched=True)
	val_data = load_data_from_json(
		os.path.join('..', 'data', 'dataset', 'dataset_eval_ground_truth.json'),
	)
	val_data = val_data.map(massage_data, batched=True)

	trainer = Seq2SeqTrainer(
		model=model,
		tokenizer=tokenizer,
		args=training_args,
		train_dataset=train_data,
		eval_dataset=val_data,
	)
	trainer.train()
	metrics = trainer.evaluate()
	print(metrics)
	return model, tokenizer, metrics

DEVICE = 'cuda'

def test_model(section, model, tokenizer):
	section_summary_results = {}
	content = section["Text"]
	section_name = section["Section"]
	ground_truth_summary = section.get("Groundtruth")[0]
	if content and ground_truth_summary:
		inputs = tokenizer(content, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
		labels = tokenizer(ground_truth_summary, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)["input_ids"]

		input_ids = inputs["input_ids"].to(DEVICE)
		attention_mask = inputs["attention_mask"].to(DEVICE)
		labels = labels.to(DEVICE)
		with torch.no_grad():
			outputs = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				labels=labels,
			)

			logits = outputs.logits
			probs = torch.softmax(logits[0], dim=-1)
			generated_ids = torch.argmax(probs, dim=-1)

			summary_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
			ground_truth_summary = tokenizer.decode(labels[0], skip_special_tokens=True)

			rouge1_f1, rouge2_f1, rougeL_f1 = calculate_rouge_scores(summary_text, ground_truth_summary)

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
					test_model(subsection, model, tokenizer)

	return section_summary_results

def calculate_rouge_scores(generated_summary, ground_truth_summary):
	scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
	scores = scorer.score(generated_summary, ground_truth_summary)
	rouge1_f1 = scores['rouge1'].fmeasure
	rouge2_f1 = scores['rouge2'].fmeasure
	rougeL_f1 = scores['rougeL'].fmeasure
	return rouge1_f1, rouge2_f1, rougeL_f1
