﻿# How to use dataset folder
This folder is used to store training, verifidation and test data. Other JPYNB files will use them.<br>

The data are stored in two formats - JSON and CSV.
CSV data is extracted from JSON data, in case someone wants to use the more straight, simpler CSV data format.

## How to Run convert_json_to_csv.py ##
*Note: This Python file is used to convert JSON data into CSV format. If you just want to use JSON format data, or want to use the existing CSV format data (train.csv, eval.csv and test.csv), you do not need to run this step.*

If you want to re-convert JSON format data into CSV format, run the following steps:
1. Check function 'convert_json_to_csv' in convert_json_to_csv.py, ensure that the JSON file names exist in this folder.
2. Run the Python file directly.
3. Three .csv files will be created. They will replace the original .csv files if the old .csv files exist.
      dataset_ground_truth.json      -----> training.csv
      dataset_eval_ground_truth.json -----> eval.csv
      dataset_test_ground_truth.json -----> test.csv

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

## Function Usage: ##

- `processing_pdf.open_file`<br>
Input: a pdf file's full path.<br> 
Return: a pyMuPDF file object, the full text of the pdf file, total pages of the pdf<br>

- ```processing_pdf.auto_find_toc```<br>
This function will get the PDF file's original table_of_content, do some cleanup to make it more neat. If the original table_of_content is missing, it will use RE mapping method to create one. However, user needs to check if the auto-generated toc is what they want. They could further customize it based on the generated template.
The purpose to do cleanup on the original toc is that some contain whitespaces (\r, \n, extra \s) making the display ugly.
Input: the pyMuPDF file object which returns from the above function `processing_pdf.open_file`<br>
Return: a composite list structure. An example is like this:<br>
```
[[1, '1 Introduction', 1],
 [1, '2 Related Work', 2],
 [1, '3 Method', 3],
 [2, '3.1 Vision Transformer (ViT)', 3],
 [2, '3.2 Fine-tuning and Higher Resolution', 4],
 [1, '4 Experiments', 4],
 [2, '4.1 Setup', 4],
 [2, '4.2 Comparison to State of the Art', 5],
 [2, '4.3 Pre-training Data Requirements', 6],
 [2, '4.4 Scaling Study', 8],
 [2, '4.5 Inspecting Vision Transformer', 8],
 [2, '4.6 Self-supervision', 8]]
 ```

 For each sub item, e.g. [2, '4.1 Setup', 4], "2" represents level 2 section (it is a sub section of "4 Experiments"), "4.1 Setup" is the title, "4" is the page. <br>
 Note: this function is only used when the PDF file has not table of content. If the PDF file has table_of_content, just call `doc.get_toc()` to get table_of_content.

 - `processing_pdf.clear_processed_folder`<br>
 Input: the folder to get cleaned up<br>
 Return: None<br>

 - `processing_pdf.find_meta_data`<br>
 Input: the pyMuPDF file object, table_of_content<br>
 Return: title, authors_info, other_info, abstract, all string type<br>

 - `processing_pdf.separate_content`<br>
 This is the main function which wraps up the PDF conversion process.<br>
 Input: whole text of pdf file, table of content<br>
 Return: a dataframe to store separated content, a dictionary which matches the output json format.<br>

 - `processing_pdf.save_dataframe`<br>
 After running this function, the dataframe is save into .csv, and the dictionary structure is saved into .json in a specified folder.<br>
 Input: the dataframe, the dictionary which are returned from the function `processing_pdf.separate_content`, the folder path to save files, the file name to save to. <br>
 Return: None<br>

 - `find_images`:<br>
 After running this function, all the images are extracted into the specified folder. Each image is named by its location in a section/subsection.<br>
 Input: pyMuPDF file object, table of content, total pages of the PDF file, folder to save to.<br>
 Note: some diagrams made from Latex could not be extracted.<br>

 ## Limitations ##
 Although arXiv papers follow a standard template, PDF structure still has much flexibility in terms of section naming, white space variance, inconsistent table of content. The functionalities of this project could not ensure a successful info extraction of all the PDF files.  
 1. Diagrams drawn by Latex are not extracted.
 2. Image might be divided when getting extracted, breaking the integrity of the original one.
 3. The caption of the image could not be extracted.
 4. Auto extracted table of content could not match the real sections completely. No regular expression could cover all those different writing styles.

## Next Step ##
1. bug fixing based on feedback
2. Extract tables
3. Image extraction bug: some overlapping images are extracted into separate files
4. Further cleanup the Latex formula and diagrams