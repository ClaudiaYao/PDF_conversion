import re 
import spacy

unicode_dict = {"\\\\N{ASTERISK OPERATOR}": "",
                "\\\\N{DAGGER}": "", 
                "\\\\N{ASTERISK OPERATOR}": "*",
                "\\\\N{LATIN SMALL LIGATURE FI}": "fi",
                "\\\\N{LATIN SMALL LIGATURE FL}": "fl",
                "\\\\N{RIGHT SINGLE QUOTATION MARK}": "'",
                "\\\\N{MULTIPLICATION SIGN}": "*",
                "\\\\N{LEFT DOUBLE QUOTATION MARK}": '"',
                "\\\\N{RIGHT DOUBLE QUOTATION MARK}": '"',
                "\\\\N{EN DASH}": "-",
                "\\\\N{ELEMENT OF}": "belong to ",
                "\\\\N{MULTIPLICATION SIGN}": "*",
                "\\\\N{GREEK SMALL LETTER BETA}": "beta",
                "\\\\N{PLUS-MINUS SIGN}": "+",
                "\\\\N{PLUS-MINUS SIGN}": "+/-",
                "\\\\N{MINUS SIGN}": "-",
                "\\\\N{ACUTE ACCENT}": ""
                }

nlp = spacy.load("en_core_web_sm") 

def clean_text(paragraph, tokenizer, lemmatizer, stopwords):
    
    text = str(paragraph.strip().encode("ascii", errors="namereplace"))
    # convert byte format to string, there is an extra chars b' at beginning and there is ' at the end. Remove them
    if text.find("b'") == 0:
        text = text [2:]
    if text[-1] == "'":
        text = text [:-1]
    # remove any references section in case Reference is not found in previous steps
    pos = -1
    if "Reference" in text:
        pos = text.index("Reference")
    elif "REFERENCE" in text:
        pos = text.index("REFERENCE")
    if pos > -1:
        text = text[: pos]

    print("ttttt:", text)
    # remove unicode chars
    for pattern, value in unicode_dict.items():
        text = re.sub(pattern, value, text) 
   
    text = text.lower()  # Lowercase words
    text = re.sub(r"-\\n", "", text)
    text = re.sub(r"\\n", " ", text)        # remove \n, \r and text
    text = re.sub(r"\\r", ' ', text)
    text = re.sub(r"\\", '', text)
    text = re.sub(r'\"', "", text)

    text = re.sub(r"([,;.])+","\g<1>", text)    # remove duplicate punct when some chars are removed
    text = re.sub(r"\[.+?\]", "", text)  # Remove [+XYZ chars] in content
    text = re.sub(r"\(.+?\)", "", text) 
    text = re.sub(r"@math\d+", "", text)
    text = re.sub(r"\bhttps*://.+@*.+\s", "", text)
    
    
    # text = re.sub(r"[0-9]+\.*[0-9]*", "", text) # remove the words with digits
    text = re.sub(r"\.\.\.", "", text)
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    # text = re.sub(r"(?<=\w)-\s*(?=\w)", "", text)  # Replace dash between words
    text = text.strip()
    # print(text, "\n")
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
