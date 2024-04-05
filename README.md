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
1. The processed PDF text and images are put into "processed" sub folder.  Suggest to call function "processing_pdf.clear_processed_folder(project_processed_data_path)" to clear the folder before each running.
2. Naming convention of the extracted images: for example, section_5_page10_6.png represents the image belongs to section 5 (matching the section 5 in the dataframe, csv and json), and it is located on page 10, 6th image captured on that page. 
3. JSON file: the field "Section_Num" matches the naming of generated image. e.g. if Section_Num is "5_1", the images in this section will be named like section_5_1_page4_1, section_5_1_page4_2.png, section_5_1_page5_3, etc.
4. The generated JSON file format is customized per requirement. It is not a standardized JSON object. Therefore, you could not convert it directly into/from the dataframe. 
5. You could use the generated dataframe directly instead of JSON or CSV. The content in dataframe is the same as that of JSON or CSV, just layout is different.
6. Create a .env file and add OPENAI_API_KEY inside

## Next Step ##
1. bug fixing based on feedback
2. Extract tables
3. Extract Title and Authors
3. Image extraction bug: some overlapping images are extracted into separate files