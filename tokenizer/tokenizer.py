from bs4 import BeautifulSoup
from track_tokenizer import parse_track_row
from suggestion_tokenizer import parse_suggestion_row
from text_tokenizer import parse_bItmH_row

tokens = []

def tokenize_row(row_raw_html: str):
    row_soup = BeautifulSoup(row_raw_html, 'html.parser')
    # Track row
    if (outer_div := row_soup.find("div", class_="tlpItem")):
        return parse_track_row(outer_div)
    # Suggestion Row
    elif (outer_div := row_soup.find("div", class_="sugTog")):
        return parse_suggestion_row(outer_div)
        
    # Text Row (multiple different meanings)
    elif (outer_div := row_soup.find("div", class_="bItmH")):
        return parse_bItmH_row(outer_div)
    else:
        return