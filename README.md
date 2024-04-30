# PDF_Conversion

## How to Run the Code

1. Clone the repository
2. Install the necessary libraries by running "pip install -r requirements.txt"
3. To run the full pipeline, run 'run_pipeline.ipynb'. Note: Provide an OpenAI API key in a .env file if you wish to use GPT functions.

## Sub Module Introduction
### processing_pdf

This module will separate long PDF document into sections and subsections based on table of content. Check README.md in the folder `processing_pdf`.

### data

This folder stores raw PDF file (paper_PDF folder), the generated training, verification and test dataset (dataset and dataset2 folders). PPT templates and generated presentation. It contains a Python file to convert JSON format to simpler, more straightforward CSV file. Check README.md in the folder `dataset`. In `dataset2` folder, a Python file `further_cleanup_json.py` is used to conduct further cleanup for the input data.

### data_preparation
This folder contains code to generate labeled datasets for training, evaluation and test. 

`prepare_data.ipynb`: generates the 3 json files under `data/dataset`.

### model_summarizer

This folder contains the code to fine-tune the LLM models. The sub folders `logs` and `results` are used to store logging files and model training results.
The file `model_training_evaluate_pipeline.jpynb` supports running both locally and on Colab. Follow the instructions in the Notebook to run and check the result.

#### bart_summarizer

`bart_summarizer.py`: contains the training and testing implementation using `BartForConditionalGeneration` and `Seq2SeqTrainer`.
`bart_summarizer.ipynb`: contains the experiment code to tune the selected hyperparameters. Note that the training results are recorded and hard-coded values are used while plotting the diagrams. This was done to save time during the experiment.
`results/bart_large_results.json`: contains the testing results including the generated text and their rouge scores.

### presentation_generation
This folder contains the code to generate presentation.

