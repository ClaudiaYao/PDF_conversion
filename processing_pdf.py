import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string
import pandas as pd
import fitz

def save_dataframe(df, save_folder, file_name):
    df.to_csv(save_folder + "/" + file_name + ".csv", index=False)
    print("save the dataframe to {}".format(save_folder + "/" + file_name + ".csv"))
    df.to_json(save_folder + "/" + file_name + ".json", orient='table',index=False)
    print("save the dataframe to {}".format(save_folder + "/" + file_name + ".json"))
    

def open_file(pdf_file):
    try:
        fitz.TOOLS.mupdf_warnings()  # empty the problem message container
        doc = fitz.open(pdf_file)
        warnings = fitz.TOOLS.mupdf_warnings()
        if warnings:
            print(warnings)
            raise RuntimeError()

        total_text = ""
        for page in doc.pages():
            total_text += page.get_text()
        return doc, total_text

    except:
        print("error when opening the pdf file {}".format(pdf_file))
        return None

    
# text parameter is a string type, which will be cleaned.
def clean_text(paragraph):

    text = str(paragraph).lower()  # Lowercase words
    text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text = re.sub(
        f"[{re.escape(string.punctuation)}]", "", text
    )  # Remove punctuation

    return text



def find_section_titles(text, title_list):
    sections_title, sections_pos = [], []
    find_pos = 0

    # find all the section titles and their positions in the text
    for index, title in enumerate(title_list):
        words = title.lower().split()
        pattern = r"{}\.?(\s|\r|\n)+".format(words[0]) + ' '.join(words[1:])
        matches = re.finditer(pattern, text)

        find_pos = None
        for match in matches:
            find_pos = match.span(0)
            break
        if find_pos is None:
            continue
        else:
            sections_title.append(title)
            sections_pos.append(find_pos)

    # find all the text belonging to that section
    sections_text = []
    total_seg = len(sections_pos)
    for i in range(total_seg):
        text_start_pos = sections_pos[i][1] + 1
        if i == 0:
            sections_text.append(text[:text_start_pos])
        elif i == total_seg - 1:
            sections_text.append(text[text_start_pos: ])
        else:
            text_end_pos = sections_pos[i+1][0]
            sections_text.append(text[text_start_pos: text_end_pos])

    return sections_title, sections_text


def get_first_level_toc(table_of_content):
    first_toc = []

    for item in table_of_content:
        if item[0] == 1:
            first_toc.append(item[1])
    return first_toc


def get_sub_toc(toc_title, table_of_content):
    sub_toc = []
    found = False
    for item in table_of_content:
        if item[1].lower() == toc_title.lower():
            found = True
            level = item[0]
            continue        
        if found:
            if item[0] == level + 1:
                sub_toc.append(item[1])
            else:
                break
    return sub_toc
            



def separate_content(text, table_of_content):
    print("starting looking for all the sections according to the provided section title info...")
    processed_content = []
    text = text.lower()

    titles = get_first_level_toc(table_of_content)
    level1_titles, level1_texts = find_section_titles(text, titles)

    for level1_index, level1_title in enumerate(level1_titles):

        sub_titles = get_sub_toc(level1_title, table_of_content)
        level2_titles, level2_texts = find_section_titles(level1_texts[level1_index], sub_titles)

        if len(level2_titles) == 0 and len(level2_texts) == 0:
            record = {"level_1": level1_title, "level_1_text": level1_texts[level1_index]}
            processed_content.append(record)
        else:    
            for level2_index, level2_title in enumerate(level2_titles):
            
                sub_titles = get_sub_toc(level2_title, table_of_content)

                if len(sub_titles) != 0:
                    level3_titles, level3_texts = find_section_titles(level2_texts[level2_index], sub_titles)


                    # did not find any sub sections, then just record all the section text
                    if len(level3_titles) == 0 and len(level3_texts) == 0:
                        record = {"level_1": level1_title, 
                                  "level_2": level2_title,
                                "level_2_content": clean_text(level2_texts[level2_index])}
                        processed_content.append(record)
                    
                    else:
                        for level3_index, level3_title in enumerate(level3_titles):

                            record = {"level_1": level1_title, "level_2": level2_title, 
                                      "level_3": level3_title,
                                    "level_3_content": clean_text(level3_texts[level3_index])}
                            processed_content.append(record)
                else:
                    record = {"level_1": level1_title, "level_2": level2_title,
                              "level_2_content": level2_texts[level2_index]}
                    processed_content.append(record)

    ds = pd.DataFrame(processed_content)
    return ds
