import nltk
# from nltk.tokenize import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string
import pandas as pd
import fitz
import json

def save_dataframe(df, json_dict, save_folder, file_name):
    full_name = save_folder + "/" + file_name + ".csv"
    df.to_csv(full_name, index=False)
    print("save the dataframe to {}".format(full_name))

    json_list = json.dumps(list(json_dict.values()))
    full_name = save_folder + "/" + file_name + ".json"
    with open(full_name, "w") as jsonfile: 
        jsonfile.write(json_list)
    print("save the dataframe to {}".format(full_name))
    

def open_file(pdf_file):
    try:
        fitz.TOOLS.mupdf_warnings()  # empty the problem message container
        doc = fitz.open(pdf_file)
        warnings = fitz.TOOLS.mupdf_warnings()
        if warnings:
            print(warnings)
            raise RuntimeError()

        total_text = ""
        total_pages = 0
        for page in doc.pages():
            total_text += page.get_text()
            total_pages += 1
        return doc, total_text, total_pages

    except:
        print("error when opening the pdf file {}".format(pdf_file))
        return None

    
# text parameter is a string type, which will be cleaned.
def clean_text(paragraph, tokenizer, lemmatizer, stopwords):

    text = str(paragraph).lower()  # Lowercase words
    text = text.encode("ascii", "ignore")
    text = text.decode()
    text = re.sub(r"\\u\.{4}", "", text)
    text = re.sub(r"\[.+?\]", "", text)  # Remove [+XYZ chars] in content
    text = re.sub(r"\n", " ", text)        # remove \n, \r and text within ()
    text = re.sub(r"\r", ' ', text)
    text = re.sub(r"\.+?\)", "", text) 
    text = re.sub(r"[0-9]+\.*[0-9]*", "", text) # remove the words with digits
    text = re.sub(r"…", "", text)  # Remove ellipsis
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    # text = re.sub(
    #     f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation

    # tokens = tokenizer(text)  # Get tokens from text
    # tokens = [t for t in tokens if not t in stopwords]  # Remove stopwords
    # tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
    # tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens

    # lemmatization
    # processed_text = ""
    # for token in tokens:
    #     processed_text = processed_text + " " + lemmatizer.lemmatize(token)
    # return processed_text
    # return ' '.join(tokens)
    return text


def get_page_sections(table_of_content, total_pages):
    page_sections = {}

    for item in table_of_content:
        page_num = item[2]
        if page_num not in page_sections: 
            page_sections[page_num] = []
        page_sections[page_num].append(item[1])

    for page_num in range(1, total_pages + 1):
        if page_num not in page_sections:
            page_sections[page_num] = [page_sections[page_num-1][-1]]
        else:
            if page_num > 1:
                page_sections[page_num].insert(0, page_sections[page_num-1][-1])
    return page_sections


def find_images(file_obj, table_of_content, total_pages, save_to_folder):

    page_sections = get_page_sections(table_of_content, total_pages)
    for page_index, page in enumerate(file_obj.pages(), start=1):
        
        # if no image on the current page, continue scanning the next page
        image_list = page.get_images(full=True)
        if len(image_list) == 0:
            continue

        # if there is only one section/sub-section, no need to check image's section location.
        cur_page_sections = page_sections[page_index]
        next_section_index = 1
        ignore_check_section = False
        if len(cur_page_sections) == 1:
            ignore_check_section = True
        else:
            # check the appearance of the next section 
            words = cur_page_sections[next_section_index].lower().split()
            pattern = r"{}\.?(\s|\r|\n)+".format(words[0]) + ' '.join(words[1:])


        blocks = page.get_text("dict")["blocks"]
        image_index = 1
        for block in blocks:
            if block['type'] == 0:
                if ignore_check_section:
                    continue
                for line in block['lines']:
                    for span in line['spans']:
                        match = re.search(pattern, span['text'].lower())
                        if match:
                            cur_page_sections = page_sections[next_section_index]
                            if next_section_index < len(cur_page_sections) - 1:
                                next_section_index += 1
                                words = cur_page_sections[next_section_index].lower().split()
                                pattern = r"{}\.?(\s|\r|\n)+".format(words[0]) + ' '.join(words[1:])
                            else:
                                ignore_check_section = True
                        
            else:
                # img_byte = block['image']     # comment this because using this could not save transparency channel images properly
                prefix = "section_" + cur_page_sections[next_section_index-1].split()[0].replace(".", "")

                ###### Xi's solution
                if image_index > len(image_list):
                    break
                img = image_list[image_index-1]
                base_image = file_obj.extract_image(img[0])
                image_bytes = base_image["image"] # binary image data
                image_extension = base_image["ext"] # not available with Pixmap
                img_file = f"{save_to_folder}/{prefix}_page{page_index}_{image_index}.{image_extension}"

                # if base_image["smask"] > 0, the image has a mask
                # the image must be enriched with transparency (alpha) bytes
                if img[1] > 0:
                    mask_image = file_obj.extract_image(img[1])
                    mask_image_bytes = mask_image["image"]
                    pix1 = fitz.Pixmap(image_bytes)        # (1) pixmap of image w/o alpha
                    mask = fitz.Pixmap(mask_image_bytes)   # (2) mask pixmap
                    pix = fitz.Pixmap(pix1, mask)          # (3) copy of pix1, image mask added  
                    image_bytes = pix.tobytes(image_extension)
                ######
                
                with open(img_file, "wb") as fh_image:
                    fh_image.write(image_bytes)
                # print("Save image to folder {}".format(img_file))
                image_index += 1


def clear_processed_folder(folder_path):
    import os, shutil

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# this function is used when the paper has not table of content
def auto_find_toc(doc):
    print("starting looking for all the sections...")
    toc = []

    sections, sub_sections = [], []
    sec_pattern_1 = re.compile(r'(\s|\n)([1-9]+\.?(\r\n|\n|\r|\s)+([A-Z][a-zA-Z-]+\s*)+)(\r|\n)')
    sec_pattern_2 = re.compile(r'(\s|\n)((I|V)+\.?(\r\n|\n|\r|\s)+([A-Z][a-zA-Z-]+\s*)+)(\r|\n)')
    subsec_pattern_1 = re.compile(r'(\b|\s|\r|\n|\r\n)([1-9]\.[1-9]+\.?(\n|\r|\s)+[A-Z][A-Za-z-]+(\s\w+)*)(\r|\n)+?')
    subsec_pattern_2 = re.compile(r'(\b|\s|\r|\n|\r\n)((I|V)+\.[A-Ea-e]+\.?(\n|\r|\s)+[A-Z][A-Za-z-]+(\s\w+)*)(\r|\n)+?')
    

    for page_index, page in enumerate(doc.pages(), start=1):
            page_text = page.get_text()
            
            while True:
                search_result = False
                match = re.search(sec_pattern_1, page_text)
                if match:
                    toc.append([1, match.group(2), page_index])
                    page_text = page_text[match.span(2)[1]:]
                    search_result = True

                match = re.search(sec_pattern_2, page_text)
                if match:
                    toc.append([1, match.group(2), page_index])
                    page_text = page_text[match.span(2)[1]:]
                    search_result = True

                match = re.search(subsec_pattern_1, page_text)
                if match:
                    toc.append([2, match.group(2), page_index])
                    page_text = page_text[match.span(2)[1]:]
                    search_result = True

                match = re.search(subsec_pattern_2, page_text)
                if match:
                    toc.append([2, match.group(2), page_index])
                    page_text = page_text[match.span(2)[1]:]
                    search_result = True 

                if search_result is False:
                    break         
    return toc

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
        if i == 0 and sections_pos[i][0] > 0:
            sections_text.append(text[:text_start_pos])
            sections_title.insert(0, "NA")
        if i == total_seg - 1:
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
    output_json_format = {}
    print("starting looking for all the sections according to the provided section title info...")
    processed_content = []
    text = text.lower()

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    titles = get_first_level_toc(table_of_content)
    level1_titles, level1_texts = find_section_titles(text, titles)

    for level1_index, level1_title in enumerate(level1_titles):

        sub_titles = get_sub_toc(level1_title, table_of_content)
        level2_titles, level2_texts = find_section_titles(level1_texts[level1_index], sub_titles)

        if len(level2_titles) == 0 and len(level2_texts) == 0:
            content = clean_text(level1_texts[level1_index],word_tokenize, lemmatizer, stop_words)
            record = {"level_1": level1_title, "level_1_content": content}
            processed_content.append(record)
            # json file requires special format
            output_json_format[level1_title] = {'Section': level1_title, "Text": content, "Subsections": [],"Groundtruth": ""}
        else:    
            for level2_index, level2_title in enumerate(level2_titles):
            
                sub_titles = get_sub_toc(level2_title, table_of_content)

                if len(sub_titles) != 0:
                    level3_titles, level3_texts = find_section_titles(level2_texts[level2_index], sub_titles)

                    # did not find any sub sections, then just record all the section text
                    if len(level3_titles) == 0 and len(level3_texts) == 0:
                        content = clean_text(level2_texts[level2_index],word_tokenize, lemmatizer, stop_words)
                        record = {"level_1": level1_title, 
                                  "level_2": level2_title,
                                "level_2_content": content}
                        processed_content.append(record)
                        

                        if level1_title not in output_json_format:
                            output_json_format[level1_title] = {'Section': level1_title, "Text": "", "Subsections": [],"Groundtruth": None}
                        
                        print(output_json_format[level1_title])
                        output_json_format[level1_title]['Subsections'].append({"Section": level2_title, "Text": content, "Subsections": [],
                        "Groundtruth": ""})
                        
                    else:
                        for level3_index, level3_title in enumerate(level3_titles):
                            content = clean_text(level3_texts[level3_index], word_tokenize, lemmatizer, stop_words)
                            record = {"level_1": level1_title, "level_2": level2_title, 
                                      "level_3": level3_title,
                                    "level_3_content": content}
                            processed_content.append(record)

                            for k, item in enumerate(output_json_format[level1_title]['Subsections']):
                                if item['Section'] == level2_title:
                                    output_json_format[level1_title]['Subsections'][k].append({"Section": level3_title, "Text": content, 
                        "Groundtruth": ""})
                else:
                    content = clean_text(level2_texts[level2_index], word_tokenize, lemmatizer, stop_words)
                    record = {"level_1": level1_title, "level_2": level2_title,
                              "level_2_content": content}
                    processed_content.append(record)

                    if level1_title not in output_json_format:
                        output_json_format[level1_title] = {'Section': level1_title, "Text": "", "Subsections": [],"Groundtruth": None}
                    output_json_format[level1_title]['Subsections'].append({"Section": level2_title, "Text": content, "Subsections": [],
                    "Groundtruth": ""})

    
    ds = pd.DataFrame(processed_content)
    return ds, output_json_format
