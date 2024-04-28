import os
import pandas as pd 
import json

# How to use this helper file to convert JSON format dataset into CSV format
# 1. check function 'convert_json_to_csv', ensure that the JSON file names exist in this folder.
# 2. Run this file directly.
# 3. Three .csv files are created. 
#      dataset_ground_truth.json      -----> training.csv
#      dataset_eval_ground_truth.json -----> eval.csv
#      dataset_test_ground_truth.json -----> test.csv

def load_data(file_path):
    with open(file_path, 'r', encoding = "utf-8") as file:
        data = json.load(file)
    return data

# Convert JSON dictionary structure to a plain dataset with only two columns - Text, and Groundtruth
def convert_dict_to_dataset(section, result):
    content = section.get("Text")
    ground_truth = section.get("Groundtruth")
    result.append({"Text": content, "Groundtruth": ground_truth})

    if "Subsections" in section and len(section['Subsections']) > 0:
        for subsection in section["Subsections"]:
            convert_dict_to_dataset(subsection, result)
    return result

def convert_json_to_csv_wrapper(json_file_full_name, converted_file_full_name):
    if not os.path.exists(json_file_full_name):
        print("the json file does not exist.")
        return

    record_list = load_data(json_file_full_name)
    result = []
    for item in record_list:
        convert_dict_to_dataset(item, result)

    data = pd.DataFrame(result)
    data.columns = ['Text', 'Groundtruth']
    data.dropna(inplace=True)
    data.to_csv(converted_file_full_name, index=False)
    print("converted to csv file - {}".format(converted_file_full_name))
    # display(data.head())


def convert_json_to_csv():
    cur_path = os.getcwd() + "/dataset"
    convert_json_to_csv_wrapper(cur_path + "/dataset_ground_truth.json", cur_path + "/training.csv")
    convert_json_to_csv_wrapper(cur_path + "/dataset_eval_ground_truth.json", cur_path + "/eval.csv")
    convert_json_to_csv_wrapper(cur_path + "/dataset_test_ground_truth.json", cur_path + "/test.csv")

convert_json_to_csv()
