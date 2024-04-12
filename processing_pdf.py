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

def save_dataframe(df, df_meta, json_dict, save_folder, file_name):
    full_name = save_folder + "/" + file_name + ".csv"
    df.to_csv(full_name, index=False)
    print("save the dataframe to {}".format(full_name))
    
    full_name = save_folder + "/" + file_name + "_meta.csv"
    df_meta.to_csv(full_name, index=False)
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
    text = re.sub(r"\[.+?\]", "", text)  # Remove [+XYZ chars] in content
    text = re.sub(r"\\n", " ", text)        # remove \n, \r and text within ()
    text = re.sub(r"\\r", ' ', text)
    text = re.sub(r"\(.*?\)", "", text) 
    text = re.sub(r"@math\d+", "", text)
    text = re.sub(r"[0-9]+\.*[0-9]*", "", text) # remove the words with digits
    text = re.sub(r"…", "", text)  # Remove ellipsis
    text = re.sub(r"\.\.\.", "", text)
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    # text = re.sub(r"(?<=\w)-\s*(?=\w)", "", text)  # Replace dash between words
    text = text.strip()
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

# get possible section/subsections on each page.
# for example, on table-of-content, it has [1, "Abstract", 1], [1, "1. Introduction", 2], [2, "1.1 Literature", 2], then on page 2, the text might be located in section "Abstract", "1. Introduction", or "1.1 Literature"
# this function is mainly used to check which section/subsection an image belongs to.
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

# find images from PDF file and save them to a folder. The code also finds the section/subsection the image belongs to.
def find_images(file_obj, table_of_content, total_pages, save_to_folder):

    page_sections = get_page_sections(table_of_content, total_pages)
    for page_index, page in enumerate(file_obj.pages(), start=1):
        
        # if no image on the current page, continue scanning the next page
        image_list = page.get_images(full=True)
        if len(image_list) == 0:
            continue

        # if there is only one section/sub-section, no need to check image's section because it definitely belongs to the only section.
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
                    text = ""
                    for span in line['spans']:
                        text += span['text']

                    match = re.search(pattern, text.lower())
                    if match:
                        cur_page_sections = page_sections[next_section_index]
                        if next_section_index < len(cur_page_sections) - 1:
                            next_section_index += 1
                            words = cur_page_sections[next_section_index].lower().split()
                            pattern = r"{}\.?(\s|\r|\n)+".format(words[0]) + ' '.join(words[1:])
                        else:
                            ignore_check_section = True
                        
            else:
                prefix = "section_" + get_section_num(cur_page_sections[next_section_index-1])

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

# each time when a new PDF file gets processed, clear up the "processed" folder
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

# find title, authors, other meta data, and abstract on the first page of the paper
# return the above information. If the information is missing, return empty string.
def find_meta_data(file_obj, table_of_content):
    
    page = next(file_obj.pages())
    blocks = page.get_text("dict")["blocks"]

    if "abstract" in table_of_content[0][1].lower():
        first_section_title = table_of_content[1][1]
    else:
        first_section_title = table_of_content[0][1]
    section_num, section_title = first_section_title.lower().split(" ", 1)

    title_block_start, authors_block_start, other_info_block_start, search_end = False, False, False, False
    title, authors_info, other_meta, abstract = "", "", [], []
    other_info = []
    for block in blocks:
        if block['type'] == 0:
            font_size = block['lines'][0]['spans'][0]['size']

            # check title
            if font_size > 11:
                if not title_block_start:
                    title = ""
                    for line in block['lines']:
                        for span in line['spans']:
                            title += span['text']
                    # sometimes, the big font is just some disturbing elements, or arXiv tags
                
                    if len(title.split()) > 6:
                        title_block_start = True
                        continue

            # both title and authors blocks have not started. The block might be header section.
            if (not title_block_start) and (not authors_block_start):
                continue
            
            # check authors info
            if  title_block_start and (not authors_block_start) and (not other_info_block_start):
                authors_block_start = True
                authors_info = ""
                for line in block['lines']:
                    for span in line['spans']:
                        authors_info += span['text']
                other_info_block_start = True
                continue
            else:
                # check the info below authors block, including the abstract section
                text = ""
                for line in block['lines']:
                    for span in line['spans']:
                        text += span['text']
                
                # have searched to the first section (usually it is 1. Introduction), stop searching
                if (section_num.lower() in text.lower()) and (section_title.lower() in text.lower()):
                    break
                other_info.append(text)

    # separate into other meta info and abstract
    other_info = ' '.join(other_info)
    if "abstract" in other_info.lower():
        pos = other_info.lower().index("abstract")
        other_meta = other_info[: pos].strip()
        abstract = other_info[pos + len("abstract"):]
    else:
        other_meta = ""
        abstract = other_info.strip()

    return title, authors_info, other_meta, abstract
                

# This function is used when the paper has not table of content. The code will use regular expression to map the table of content. However, it is better to manually check the generated table of content and then do some customization based on the auto-generated result.
def auto_find_toc(doc):
    print("starting looking for all the sections...")
    toc = []

    sec_pattern_1 = re.compile(r'(\s|\n)([1-9]+\.?(\r\n|\n|\r|\s)+([A-Z][a-zA-Z-]+\s*)+)(\r|\n)')
    sec_pattern_2 = re.compile(r'(\s|\n)((I|V)+\.?(\r\n|\n|\r|\s)+([A-Z][a-zA-Z-]+\s*)+)(\r|\n)')
    subsec_pattern_1 = re.compile(r'(\b|\s|\r|\n|\r\n)([1-9]\.[1-9]+\.?(\n|\r|\s)+[A-Z][A-Za-z-]+(\s\w+)*)(\r|\n)+?')
    subsec_pattern_2 = re.compile(r'(\r|\n)([A-E]\.(\n|\r|\s)+([a-zA-Z-]+\s)+)')
    
    # Check each page and extract the text pattern which indicates it is section/subsection. The generated table-of-content follows the same format as the puMuPDF's generated one.
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

# Analyze the given text and given sub-section titles, separate the text into sub-sections.
def find_section_titles(text, title_list):
    sections_title, sections_pos = [], []
    find_pos = 0

    # find all the section titles and their positions in the text
    for _, title in enumerate(title_list):
        words = title.lower().split(" ", 1)
        if len(words) > 1:
            pattern = r"{}\.?(\s|\r|\n)+".format(words[0]) + words[1]
            matches = re.finditer(pattern, text.lower())
        else:
            # Some sections (e.g. Abstract) without section number. To avoid finding the same words instead of section tile, keep the original capital
            pattern = title
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
            sections_text.insert(0, text[:sections_pos[i][0]])
            sections_title.insert(0, "No_title")

        if i == total_seg - 1:
            # usually the last part of the paper contains references. Cut out the last part
            left_text = text[text_start_pos:]
            if "References" in left_text:
                left_text = left_text[:left_text.index("References")]
            elif "REFERENCES" in left_text:
                left_text = left_text[:left_text.index("REFERENCES")]
            sections_text.append(left_text)
        else:
            text_end_pos = sections_pos[i+1][0]
            sections_text.append(text[text_start_pos: text_end_pos])
    
    return sections_title, sections_text

# get the first level section titles of table-of-content
def get_first_level_toc(table_of_content):
    first_toc = []

    for item in table_of_content:
        if item[0] == 1:
            first_toc.append(item[1])
    return first_toc

# get the subsection titles of a specified section toc_title
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
            
# Extract the section_num. For example, a section in talbe-of-content shows [2, "2.3 Experiment result", 3], the section_num should be "2_3". This result is used when naming images which belong to this section.
def get_section_num(section_title):
    section_num = section_title.split()[0]
    if section_num[-1] == ".":
        section_num = section_num[:-1]
    return section_num.replace(".", "_")

# set up the json file field structure
def build_initial_output_json(table_of_content):
    json_dict = {}
    titles = get_first_level_toc(table_of_content)
    if titles[0].lower() != "abstract":
        titles.insert(0, "Abstract")

    for i, title in enumerate(titles):              
        json_dict[title] = {"Section_Num": get_section_num(title), 'Section': title, "Text": "", "Subsections": [],"Groundtruth": ""}

    return json_dict

def clean_section_title(title):
    res = re.sub(r"\n", "", title)
    res = re.sub(r"\r", "", res)
    res = re.sub(r"\s+", " ", res)
    res = res.strip()
    return res


# This is the main function to separate the content of a PDF into multiple section/subsections, extract images from the PDF file and naming them based on which sections they belong to.
def separate_content(text, table_of_content):
    print("starting looking for all the sections according to the provided section title info...")
    output_json_format = build_initial_output_json(table_of_content)
    processed_content = []
    text = text.lower()

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    titles = get_first_level_toc(table_of_content)
    # this function removes the appendix and reference section, so the returned level1_titles and level1_texts will not contain the two categories
    level1_titles, level1_texts = find_section_titles(text, titles)

    for level1_index, level1_title in enumerate(level1_titles):
        # We will not record the information in Reference or Appendix sections
        if "Reference" in level1_title or "REFERENCE" in level1_title or \
        "Appendix" in level1_title or "APPENDIX" in level1_title:
            continue

        if level1_index == 0 and level1_title == "No_title":
            content = clean_text(level1_texts[0], word_tokenize, lemmatizer, stop_words)
            record = {"level_1": "Abstract", "level_1_content": content}
            processed_content.append(record)
            output_json_format["Abstract"]["Text"] = content
            continue


        sub_titles = get_sub_toc(level1_title, table_of_content)
        level2_titles, level2_texts = find_section_titles(level1_texts[level1_index], sub_titles)

        if len(level2_titles) == 0 and len(level2_texts) == 0:
            content = clean_text(level1_texts[level1_index], word_tokenize, lemmatizer, stop_words)
            level1_title = clean_section_title(level1_title)
            record = {"level_1": level1_title, "level_1_content": content}
            processed_content.append(record)
            # json file requires special format
            
            output_json_format[level1_title] = {"Section_Num": get_section_num(level1_title), 'Section': level1_title, "Text": content, "Subsections": [],"Groundtruth": ""}
        else:    
            for level2_index, level2_title in enumerate(level2_titles):
                # this operation will put the abstract at the end of the
                if level2_index == 0 and level2_title == "No_title":
                    output_json_format[level1_title]['Text'] = clean_text(level2_texts[0],word_tokenize, lemmatizer, stop_words)
                    continue

                sub_titles = get_sub_toc(level2_title, table_of_content)

                if len(sub_titles) != 0:
                    level3_titles, level3_texts = find_section_titles(level2_texts[level2_index], sub_titles)

                    # did not find any sub sections, then just record all the section text
                    if len(level3_titles) == 0 and len(level3_texts) == 0:
                        content = clean_text(level2_texts[level2_index],word_tokenize, lemmatizer, stop_words)
                        level2_title = clean_section_title(level2_title)
                        record = {"level_1": level1_title, 
                                  "level_2": level2_title,
                                "level_2_content": content}
                        processed_content.append(record)
                        
                        output_json_format[level1_title]['Subsections'].append({"Section_Num": get_section_num(level2_title) , "Section": level2_title, "Text": content, "Subsections": [],
                        "Groundtruth": ""})
                        
                    else:
                        for level3_index, level3_title in enumerate(level3_titles):
                            content = clean_text(level3_texts[level3_index], word_tokenize, lemmatizer, stop_words)
                            level3_title = clean_section_title(level3_title)
                            record = {"level_1": level1_title, "level_2": level2_title, 
                                      "level_3": level3_title,
                                    "level_3_content": content}
                            processed_content.append(record)
                            
                            
                            for k, item in enumerate(output_json_format[level1_title]['Subsections']):
                                if item['Section'] == level2_title:
                                    if level3_title == "No_title":
                                        output_json_format[level1_title]['Subsections'][k]['Text'] = content
                                    else:
                                        output_json_format[level1_title]['Subsections'][k].append({"Section_Num": get_section_num(level3_title), "Section": level3_title, "Text": content, "Groundtruth": ""})
                else:
                    content = clean_text(level2_texts[level2_index], word_tokenize, lemmatizer, stop_words)
                    level2_title = clean_section_title(level2_title)
                    record = {"level_1": level1_title, "level_2": level2_title,
                              "level_2_content": content}
                    processed_content.append(record)

                    output_json_format[level1_title]['Subsections'].append({"Section_Num": get_section_num(level2_title), "Section": level2_title, "Text": content, "Subsections": [],
                    "Groundtruth": ""})

    
    ds = pd.DataFrame(processed_content)
    return ds, output_json_format
