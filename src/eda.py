import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import DataLoader      
pd.set_option("display.max_columns", None)

loader = DataLoader()

df_base = loader.load_data("data/processed/temiz_veri.csv")

df = df_base.copy()

def analyze_projects_by_year(df: pd.DataFrame) -> pd.Series:
    project_counts = df.groupby("year").size().sort_index()

    plt.plot(project_counts.index, project_counts.values)
    plt.xlabel("Year")
    plt.ylabel("Number of Projects")
    plt.title("ISEF Projects by Year")
    plt.show()

    return project_counts


def analyze_categories(df: pd.DataFrame) -> pd.Series:
    category_count = df.groupby("category", dropna = False).size().sort_values()

    sns.barplot(x= category_count.values,y = category_count.index)
    plt.xlabel("Project Count")
    plt.ylabel("Category")
    plt.title("ISEF Projects by Categories")
    plt.show()

    return category_count


def analyze_category_trends(df: pd.DataFrame):
    tab = pd.crosstab(df["year"],df["category"], normalize = True)

    sns.heatmap(tab)
    plt.xlabel("Category")
    plt.ylabel("Year")
    plt.title("ISEF Project Categories by Year")
    plt.show()

    return tab


def analyze_missingness(df):
    nan_count = df.isnull().sum()

    sns.barplot(x = nan_count, y = df.columns) 
    plt.xlabel("NaN Counts")
    plt.ylabel("Columns")
    plt.title("NaN Counts For Each Column")
    plt.show()

    return nan_count


def count_abstract(val):
    if pd.isna(val):
        return np.nan
    return len(val.split())


def analyze_abstract_length(df: pd.DataFrame):
    count = df["abstract"].apply(count_abstract)

    sns.histplot(count.dropna())
    plt.xlabel("Word Count")
    plt.ylabel("Number of Abstracts")
    plt.title("Distribution of Abstract Length")
    plt.show()

    return count
    

def analyze_abstract_length_by_year(df: pd.DataFrame):
    ab_count = df["abstract"].apply(count_abstract)
    result = ab_count.groupby(df["year"]).median().sort_index()

    plt.plot(result.index, result.values)
    plt.xlabel("Year")
    plt.ylabel(" Median of the Length of Abstraction Section")
    plt.show()

    return result
    

def analyze_text_availability(df: pd.DataFrame):
    data = {"total_projects":len(df),"project_without_ab":df["abstract"].isnull().sum(),
            "project_with_ab":(len(df)) - (df["abstract"].isnull().sum()),
            "prct_with_ab": ((len(df)) - (df["abstract"].isnull().sum())) / (len(df)) }
    
    s = pd.Series(data)
    
    return s