# PDF_conversion

## How to Run the Code ##
1. Clone the repository
2. Install the necessary libraries by running "pip install -r requirements.txt"
3. In VSCode, Open processing_pdf.ipynb.
4. Run the cells in sequence.

## Customization ##
1. You could choose any pdf file, but you need to put it in "data" sub folder.
2. The program will extract table_of_content from the pdf file automatically. However, if you just want to extract partial sections of the pdf, you could edit table_of_content to remove some sections.
3. The converted text result shows in dataframe "ds" immediately. It also saves text to .csv and .json files to sub folder "processed" with the same name as the original pdf. Images are saved to sub folder "processed", too.

## Note ##
1. The processed PDF text and images are put into "processed" sub folder.  Suggest to call function "processing_pdf.clear_processed_folder(project_processed_data_path)" to clear its content before each running.
2. Naming convention of the extracted images: for example, section5_page10_6.jpg represents the image belongs to section 5 (matching the section 5 in the dataframe, csv and json), and it is located on page 10, 6th image captured on that page. 

## Next Step ##
1. bug fixing based on feedback
2. Extract tables 
3. Image extraction bug: some overlapping images are extracted into separate files