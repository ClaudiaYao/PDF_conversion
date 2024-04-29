# PDF_Conversion

## How to Run the Code

1. Clone the repository
2. Install the necessary libraries by running "pip install -r requirements.txt"
3. To run the full pipeline, run 'run_pipeline.ipynb'. Note: Provide an OpenAI API key in a .env file if you wish to use GPT functions.
4.

## Process PDF Module

This module will separate long PDF document into sections and subsections based on table of content. Check README.md in the folder `processing_pdf`.

## Datset Folder

This folder stores training, verification and test data. It also contains a Python file to convert JSON format to simpler, more straightforward CSV file. Check README.md in the folder `dataset`.

## Paper_pdf Folder

This folder contains some sample PDF files which will be used by folder `Process PDF`. Please note just a small portion of training PDF files are put inside this folder.
