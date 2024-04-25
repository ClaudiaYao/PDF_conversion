import os
import json
from transformers import LEDForConditionalGeneration, LEDTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from rouge_score import rouge_scorer


def get_model(model_name):
	model = LEDForConditionalGeneration.from_pretrained(model_name)
	tokenizer = LEDTokenizer.from_pretrained(model_name)
	return model, tokenizer

def load_data(file_path):
	with open(file_path, 'r',encoding='utf-8') as file:
		data = json.load(file)
	return data

def massage_data(tokenizer, sections):
	results = []
	for section in sections:
		content = section.get('Text')
		ground_truth = section.get('Groundtruth')
		if content and ground_truth:
			inputs = tokenizer(content, padding='max_length', return_tensors='pt')
			input_ids = inputs.input_ids.to('cuda')
			labels = tokenizer(ground_truth, padding='max_length', return_tensors='pt').input_ids.to('cuda')
			
			# Get attention mask
			attention_mask = inputs.attention_mask
			# Add to results
			results.append({
				'input_ids': input_ids, 
				'attention_mask': attention_mask, 
				'labels': labels,
			})
			
		# Process the subsections if they exist
		subsections = section.get('Subsections')
		if type(subsections) == list:
			results += massage_data(tokenizer, subsections)
	return results

def get_training_args():
	return Seq2SeqTrainingArguments(
		output_dir='./Checkpoints',
		num_train_epochs=4,
		per_device_train_batch_size=8,
		save_steps=10_000,
		eval_steps=5_000,
		logging_steps=100,
		warmup_steps=500,
		label_smoothing_factor=0.1,
		predict_with_generate=True,
		fp16=True,
		gradient_accumulation_steps=2, 
	)

def train_model(model_name):
	model, tokenizer = get_model(model_name)
	training_args = get_training_args()

	train_data = load_data(
		os.path.join('..', 'dataset', 'dataset_ground_truth.json'),  # 100 pdfs
	)
	train_data = massage_data(tokenizer, train_data)
	test_data = load_data(
		os.path.join('..', 'dataset', 'dataset_test_ground_truth.json'),   #20 pdfs
	)
	test_data = massage_data(tokenizer, test_data)
	val_data = load_data(
		os.path.join('..', 'dataset', 'dataset_eval_ground_truth.json'),  #20 pdfs
	)
	val_data = massage_data(tokenizer, val_data)

	trainer = Seq2SeqTrainer(
		model=model,
		tokenizer=tokenizer,
		args=training_args,
		train_dataset=train_data,
		eval_dataset=val_data,
	)
	trainer.train()
	metrics = trainer.evaluate()
	print("Evaluation Loss:", metrics["loss"])
	print("Rouge Score:", metrics["rouge1"])


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