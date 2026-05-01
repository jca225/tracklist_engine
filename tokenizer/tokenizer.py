from bs4 import BeautifulSoup

from ._parser import BS_PARSER
from .track_tokenizer import parse_track_row
from .suggestion_tokenizer import parse_suggestion_row
from .text_tokenizer import parse_bItmH_row


def tokenize_row(row_raw_html: str):
    """Dispatch a raw HTML row to the right parser based on its outer container.

    Each parser has its own signature contract:
      - parse_track_row: raw HTML string (re-parses internally)
      - parse_suggestion_row: bs4.Tag
      - parse_bItmH_row: raw HTML string
    """
    row_soup = BeautifulSoup(row_raw_html, BS_PARSER)
    if (outer_div := row_soup.find("div", class_="tlpItem")):
        return parse_track_row(str(outer_div))
    if (outer_div := row_soup.find("div", class_="sugTog")):
        return parse_suggestion_row(outer_div)
    if (outer_div := row_soup.find("div", class_="bItmH")):
        return parse_bItmH_row(str(outer_div))
    return None


def classify_row(row_raw_html: str) -> str:
    """Return a string label for the row kind without fully parsing it."""
    row_soup = BeautifulSoup(row_raw_html, BS_PARSER)
    if row_soup.find("div", class_="tlpItem"):
        return "track"
    if row_soup.find("div", class_="sugTog"):
        return "suggestion"
    if row_soup.find("div", class_="bItmH"):
        return "text"
    if row_soup.find(id="playerWidget"):
        return "player_widget"
    if row_soup.find(id="tl_save"):
        return "save_footer"
    return "unknown"
