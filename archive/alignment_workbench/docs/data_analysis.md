# Data Science EDA & Pipeline Organization Guide

Jupyter Notebooks are excellent for rapid exploration but can quickly turn into unreadable "spaghetti code." To prevent this, follow these industry-standard practices for structuring your notebooks, organizing your project, and solidifying your pipeline.

## Phase 1: Structuring a Single EDA Notebook
Treat your `.ipynb` file like a technical report. Use Markdown cells to explain *why* you are doing something, not just *what* the code does. Follow this top-to-bottom flow:

1. **Title and Objective (Markdown):** State the goal of this specific notebook (e.g., "Analyze customer churn drivers").
2. **Imports and Config (Code):** Place *all* imports (`pandas`, `matplotlib`, etc.) in the very first cell. Set visual configurations (like `sns.set_theme()`) here as well.
3. **Data Loading (Code):** Load the raw data. Print the shape and the first few rows.
4. **Data Sanity Checks (Code):** Check for missing values (`.isna().sum()`), duplicates, and basic data types (`.info()`). 
5. **Univariate Analysis:** Look at one variable at a time (Histograms for numericals, countplots for categoricals). *Write Markdown conclusions after your plots.*
6. **Bivariate/Multivariate Analysis:** Look at relationships between variables (Scatter plots, correlation matrices). 
7. **Summary & Next Steps (Markdown):** At the bottom, write a summary of your findings and what needs to be done next.

---

## Phase 2: Organizing the Project Directory
Do not put everything in one giant notebook. Break your work into sequentially numbered notebooks. Follow a structure similar to the **Cookiecutter Data Science** standard:

\`\`\`text
my_project/
│
├── data/
│   ├── raw/               <- NEVER modify this data. Read-only.
│   └── processed/         <- Cleaned data saved from your notebooks/scripts.
│
├── notebooks/             <- Where your EDA lives
│   ├── 01-data-cleaning.ipynb
│   ├── 02-exploratory-analysis.ipynb
│   └── 03-feature-engineering.ipynb
│
└── src/                   <- The actual "Pipeline" code (Python files)
    ├── __init__.py
    ├── data_cleaning.py   <- Functions moved out of notebooks
    └── plotting.py
\`\`\`

---

## Phase 3: Solidifying the Pipeline (Notebook to Production)
To transition from messy EDA to a solid pipeline, aggressively move code **out** of your notebook and into `.py` files. 

**The Rule of Three:** *If you write the same chunk of code three times, or if a data transformation takes more than 5 lines of code, it belongs in a function in a `.py` file.*

### The Workflow:
1. **Draft in the Notebook:** Write your messy pandas code to clean a column or engineer a feature.
2. **Refactor into a Function:** Once it works, wrap that code in a Python function right there in the notebook. 
   \`\`\`python
   def clean_price_column(df):
       df['price'] = df['price'].str.replace('$', '').astype(float)
       return df
   \`\`\`
3. **Evict to a `.py` File:** Cut that function out of your notebook, open your `src/data_cleaning.py` file, and paste it there.
4. **Import and Rerun:** Go back to your notebook, import the function from your own module, and use it. To ensure your notebook always uses the latest version of your `.py` files without needing to restart the kernel, use the `autoreload` magic commands at the top of your notebook:
   
   \`\`\`python
   %load_ext autoreload
   %autoreload 2
   
   from src.data_cleaning import clean_price_column
   
   df = clean_price_column(df)
   \`\`\`

By doing this, your notebook transforms from a messy script into a clean report that calls reliable, tested functions. You can later string these `.py` functions together into a formal pipeline (like a scikit-learn `Pipeline` or an Airflow DAG).