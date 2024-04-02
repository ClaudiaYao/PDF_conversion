# PDF_conversion

## How to Run the Code ##
1. Clone the repository
2. Install the necessary libraries by running "pip install -r requirements.txt"
3. In VSCode, Open processing_pdf.ipynb.
4. Run the cells in sequence.

## Customization ##
1. You could choose any pdf file, but you need to put it in "data" sub folder.
2. The program will extract table_of_content from the pdf file automatically. However, if you just want to extract partial sections of the pdf, you could edit table_of_content to remove some sections.
3. The converted result shows in dataframe "ds" immediately. It also saves to .csv and .json files in sub folder "processed" with the same name as the original pdf.

## Next Step ##
1. handle the image