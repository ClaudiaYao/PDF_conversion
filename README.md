# PDF_Conversion

## How to Run the Code

1. Clone the repository
2. Install the necessary libraries by running "pip install -r requirements.txt"
3. To run the full pipeline, run 'run_pipeline.ipynb'. Note: Provide an OpenAI API key in a .env file if you wish to use GPT functions.

## Sub Module Introduction

### processing_pdf

This module will separate long PDF document into sections and subsections based on table of content. Check README.md in the folder `processing_pdf`.

### data

This folder stores raw PDF file (paper_PDF folder), the generated training, verification and test dataset (dataset and dataset2 folders). PPT templates and generated presentation. <br>
It contains a Python file to convert JSON format to simpler, more straightforward CSV file. Check README.md in the folder `dataset`. <br>
In `dataset2` folder, a Python file `further_cleanup_json.py` is used to conduct further cleanup for the input data.

### data_preparation

This folder contains code to generate labeled datasets for training, evaluation and test.

`prepare_data.ipynb`: generates the 3 json files under `data/dataset`.

### model_summarizer

This folder contains the code to fine-tune the LLM models. The sub folders `logs` and `results` are used to store logging files and model training results.

#### allenai_summarizer

`allenai_summarizer.py`: contains the code to define the model, pre-processing functions and functions to train, evaluate and test the model.

`allenai_LED_model_training.ipynb` : Colab notebook to see the plots of training and validation loss and the Rogue scores. The results of the evaluation using the test set are available in the notebook.

`allenai_LED_model_inference.ipynb` : Colab notebook with the link to download the model checkpoint. The LED model is loaded from the checkpoint in this notebook and the inference using the LED model using a pdf is  shown in the output.

`model_training_evaluate_pipeline.jpynb` - supports running both locally and on Colab. 

Follow the instructions in the notebook to run and check the result.


#### bart_summarizer

`bart_summarizer.py`: contains the training and testing implementation using `BartForConditionalGeneration` and `Seq2SeqTrainer`.<br>
`bart_summarizer.ipynb`: contains the experiment code to tune the selected hyperparameters. Note that the training results are recorded and hard-coded values are used while plotting the diagrams. This was done to save time during the experiment.<br>
`results/bart_large_results.json`: contains the testing results including the generated text and their rouge scores.

### presentation_generation

This folder contains the code to generate the final PowerPoint presentation. Check README.md in the folder `presentation_generation`.
