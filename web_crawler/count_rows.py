import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import Counter, defaultdict
# Count DJ sets in DB (and html files if you want)
import sqlite3
from pathlib import Path

# Small function to count the rows in our db
DB_PATH = Path("/home/ubuntu/tracklist_engine/data/db/music_database.db")


conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM dj_sets")
print("dj_sets_count:", cur.fetchone()[0])
conn.close()

