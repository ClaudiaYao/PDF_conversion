import json
import os
from transformers import BartForConditionalGeneration, BartTokenizer, LEDForConditionalGeneration, LEDTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
from rouge_score import rouge_scorer
import torch
import textwrap

MAX_LENGTH = 477


def get_model(model_name):
	# model = LEDForConditionalGeneration.from_pretrained(model_name)
	# tokenizer = LEDTokenizer.from_pretrained(model_name)
	model = BartForConditionalGeneration.from_pretrained(model_name)
	tokenizer = BartTokenizer.from_pretrained(model_name)
	return model, tokenizer

def load_data(file_path, split):
	data = load_dataset('json', data_files=file_path, split=split)
	return data

def load_json(file_path):
    with open(file_path, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def get_data_massager(tokenizer):
	def preprocess_function(sections):
		"""
		Function to preprocess data from JSON format, handling subsections.

		Args:
				examples: A dictionary containing "Text", "Groundtruth", and "Subsections" keys.

		Returns:
				A dictionary containing processed tensors for model input.
		"""
		# Concatenate text from subsections
		# if sections["Subsections"]:
		# 	subsection_text = " ".join([subsection["Text"] for subsection in sections["Subsections"]])
		# 	sections["Text"] = sections["Text"] + " " + subsection_text

		# inputs = tokenizer(sections["Text"], return_tensors="pt", padding=True, truncation=True, max_length=128)
		# labels = tokenizer(sections["Groundtruth"], return_tensors="pt", padding=True, truncation=True, max_length=128)
		inputs = tokenizer(sections["Text"], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
		labels = tokenizer(sections["Groundtruth"], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)

		return {
			"input_ids": inputs.input_ids.to('cuda'),
			"attention_mask": inputs.attention_mask.to('cuda'),
			"labels": labels.input_ids.to('cuda'),
		}
	return preprocess_function


# def massage_data(tokenizer, sections):
# 	results = []
# 	for section in sections:
# 		content = section.get('Text')
# 		ground_truth = section.get('Groundtruth')
# 		if content and ground_truth:
# 			inputs = tokenizer(content, return_tensors='pt', padding=True, truncation=True, max_length=256)
# 			input_ids = inputs.input_ids.to('cuda')
# 			labels = tokenizer(ground_truth, return_tensors='pt', padding=True, truncation=True, max_length=256).input_ids.to('cuda')
			
# 			# Get attention mask
# 			attention_mask = inputs.attention_mask.to('cuda')
# 			# Add to results
# 			results.append({
# 				'input_ids': input_ids, 
# 				'attention_mask': attention_mask, 
# 				'labels': labels,
# 			})
			
# 		# Process the subsections if they exist
# 		subsections = section.get('Subsections')
# 		if type(subsections) == list:
# 			results += massage_data(tokenizer, subsections)
# 	return results

def get_training_args():
	return Seq2SeqTrainingArguments(
		output_dir='./Checkpoints',
		num_train_epochs=3,
		per_device_train_batch_size=8,
		save_strategy='epoch',
		# save_steps=10_000,
		eval_steps=5_000,
		logging_steps=100,
		warmup_steps=500,
		label_smoothing_factor=0.1,
		predict_with_generate=True,
		learning_rate=0.01,
		fp16=True,
		gradient_accumulation_steps=2, 
	)

def train_model(model_name):
	model, tokenizer = get_model(model_name)
	training_args = get_training_args()

	massage_data = get_data_massager(tokenizer)

	train_data = load_data(
		os.path.join('..', 'dataset', 'dataset_ground_truth.json'),  # 100 pdfs
		split='train',
	)
	train_data = train_data.map(massage_data, batched=True)
	val_data = load_data(
		os.path.join('..', 'dataset', 'dataset_eval_ground_truth.json'),  #20 pdfs
		split='train',
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
	# print("Evaluation Loss:", metrics["loss"])
	# print("Rouge Score:", metrics["rouge1"])
	return model, tokenizer

DEVICE = 'cuda'

def test_model(section, model, tokenizer):
	section_summary_results = {}
	content = section["Text"]
	section_name = section["Section"]
	ground_truth_summary = section.get("Groundtruth")[0]
	if content and ground_truth_summary:
		# Tokenize the content

		inputs = tokenizer(content, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
		labels = tokenizer(ground_truth_summary, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)["input_ids"]

		input_ids = inputs["input_ids"].to(DEVICE)
		attention_mask = inputs["attention_mask"].to(DEVICE)
		labels = labels.to(DEVICE)
		with torch.no_grad():
			outputs = model.generate(
				input_ids=input_ids,
				# num_beams=5,
				# no_repeat_ngram_size=2,
			)

			# Decode the generated summary using the tokenizer
			summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
			ground_truth_summary = tokenizer.decode(labels[0], skip_special_tokens=True)

				# Calculate ROUGE scores
			# rouge1_f1, rouge2_f1, rougeL_f1 = model_summarizer.calculate_rouge_scores(summary_text, ground_truth_summary)

		section_summary_results["Section Name"] = section_name
		section_summary_results["Generated Summary"] = summary_text
		# section_summary_results["ROUGE-1 F1"] = rouge1_f1
		# section_summary_results["ROUGE-2 F1"] = rouge2_f1
		# section_summary_results["ROUGE-L F1"] = rougeL_f1
		
		print("Section Name: ", section_name)
		wrapped_output = textwrap.fill(str(summary_text), width=80)
		print("Generated Summary: ", wrapped_output)

		if "Subsections" in section:
				for subsection in section["Subsections"]:
					test_model(subsection, model, tokenizer)

	return section_summary_results

# #### old ###


# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class SummarizationModel(nn.Module):
# 	def __init__(self, pretrained_model):
# 		super().__init__()
# 		self.pretrained_model = pretrained_model 

# 	def forward(self, input_ids, attention_mask):
# 		outputs = self.model(
# 			input_ids=input_ids,
# 			attention_mask=self.attention_mask,
# 			labels=self.labels,
# 		)
# 		return outputs


# def calculate_rouge_scores(generated_summary, ground_truth_summary):
# 	scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
# 	scores = scorer.score(generated_summary, ground_truth_summary)
# 	rouge1_f1 = scores['rouge1'].fmeasure
# 	rouge2_f1 = scores['rouge2'].fmeasure
# 	rougeL_f1 = scores['rougeL'].fmeasure
# 	return rouge1_f1, rouge2_f1, rougeL_f1

# def tokenize_sections(tokenizer, section):
# 	seq_length = 1024
# 	content = section.get("Text")
# 	ground_truth = section.get("Groundtruth")
# 	results = []
# 	if content and ground_truth:
# 		inputs = tokenizer(content, return_tensors="pt", max_length=seq_length, truncation=True)
# 		labels = tokenizer(ground_truth, return_tensors="pt", max_length=seq_length, truncation=True)["input_ids"]
		
# 		# Get attention mask
# 		input_ids = inputs["input_ids"]
# 		attention_mask = inputs["attention_mask"]
# 		# Add to results
# 		results.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
		
# 	# Process the subsections if they exist
# 	if "Subsections" in section:
# 		for subsection in section["Subsections"]:
# 			results += tokenize_sections(tokenizer, subsection)
# 	return results

# # Function to train model
# def train_model(model, tokenizer, train_loader):
# 	total_loss = 0.0  # Initialize total loss
# 	for section in train_loader:
# 		results = []
# 		tokenize_sections(tokenizer, section)
# 		for result in results:
# 			inputs = result["input_ids"]
# 			attention_mask = result["attention_mask"]
# 			labels = result["labels"]
# 			output = model.forward(inputs, attention_mask, labels)

# 		# Backward pass
# 			self.optimizer.zero_grad()
# 			loss.backward()
# 			self.optimizer.step()
# 			total_loss += loss.item()
# 	return total_loss


# def train():
# 	model_name = "allenai/led-large-16384-arxiv"
# 	model = SummarizationModel(
# 		model_name=model_name,
# 		attention_mask=
# 	)