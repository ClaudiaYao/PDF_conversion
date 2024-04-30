import os
import pandas as pd 
import json
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from langdetect import detect


# How to use this helper file to further cleanup JSON format dataset
# 1. check function 'cleanup_all' which is located at the end of this file, ensure that the JSON file names exist in this folder.
# 2. Run this file directly.
# 3. Three new json files are created. The cleanup depends on the customization in the function `cleanup`.
# Currently, the cleanup includes: 
#      remove those non-English content
#      remove those sentences including greek characters.
#      remove those sentences including more than 2 single chars.
#      remove those sentences including more than 2 digits.
#      remove those sentences including more than 2 single-digit/single-char words.
#      remove those sentences including curly braces {some text}
#      remove those sentences where digits account for more than 50% of total words
#      Further cleanup the punctuation.

def load_data(file_path):
    with open(file_path, 'r', encoding = "utf-8") as file:
        data = json.load(file)
    return data

def cleanup(content):
    content = content.lower()
    if content == "":
        return ""
    try:
        if detect(content) != "en":
            return ""
    except:
        pass

    # sentences = sent_tokenize(content)
    sentences = content.split(".")
    for i in range(len(sentences)-1, -1, -1):
        sentences[i] = re.sub(r" {latin small ligature ff}", "ff", sentences[i])
        sentences[i] = re.sub(r" {latin small ligature ffi}", "ff", sentences[i])
        if "{greek" in sentences[i]:
            del sentences[i]
            continue 
        
        to_delete = False
        total_digits, single_letter = 0, 0
        total_words = 0
        for word in word_tokenize(sentences[i]):
            total_words += 1
            if word.isdigit():
                total_digits += 1
            if len(word) > 1 and word[0]=="-" and word[1:].isdigit():
                total_digits += 1
            if len(word) == 1 and word != "i" and word != "a":
                single_letter += 1
            if len(re.findall(r"{.+}", sentences[i])) > 0:
                to_delete = True
                break

        if to_delete is True or total_digits >2 or single_letter >2 or (total_words > 0 and total_digits / total_words >= 0.5) or (total_words > 0 and single_letter / total_words >= 0.5):
            del sentences[i]
        
    result = ". ".join(sentences)
    result = re.sub(r" \. ", "", result)
    result = re.sub(r" , ", "", result)
    result = re.sub(r"\w\.", "", result)
    result = re.sub(r"\d\.", "", result)
    result = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "", result)
    return result
            
# recursively find all the sections/subsections and replace its content with cleaned up text    
def update_text(section):
    content = section.get("Text")
    section["Text"] = cleanup(content)

    if "Subsections" in section and len(section['Subsections']) > 0:
        for subsection in section["Subsections"]:
            update_text(subsection)
    return

def cleanup_wrapper(json_file_full_name, converted_file_full_name):
    if not os.path.exists(json_file_full_name):
        print("the json file does not exist.")
        return

    record_list = load_data(json_file_full_name)
    for item in record_list:
        update_text(item)

    with open(converted_file_full_name, "w") as jsonfile: 
        jsonfile.write(json.dumps(record_list))
    print("save the dataframe to {}".format(converted_file_full_name))


def cleanup_all():

    cur_path = os.getcwd() + "/data/dataset2"
    cleanup_wrapper(cur_path + "/dataset3_ground_truth.json", cur_path + "/dataset4.json")
    cleanup_wrapper(cur_path + "/eval3_ground_truth.json", cur_path + "/eval4.json")
    cleanup_wrapper(cur_path + "/test3_ground_truth.json", cur_path + "/test4.json")


cleanup_all()