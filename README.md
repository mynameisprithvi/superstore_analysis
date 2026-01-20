# Superstore Sales & Profitability Analysis (EDA + Statistics)

## Objective
This project analyzes sales and profitability patterns in a retail superstore dataset
to understand which factors drive profit and loss. The analysis focuses on identifying
structural relationships between sales, discounts, product categories, and customer
segments, supported by statistical testing.

## Key Questions
- How do sales and profit vary across product categories and customer segments?
- What is the relationship between discount levels and profitability?
- Are observed differences in profitability across segments statistically significant?
- Do high sales volumes necessarily correspond to high profits?

## Dataset
The dataset contains transactional-level retail data including sales, profit, discount,
product category, customer segment, and geographic information. The data is observational
and reflects historical retail performance rather than controlled experiments.

## Methodology
- Exploratory Data Analysis (EDA) to examine distributions, trends, and group-level differences
- Aggregation and visualization of sales and profit by category, segment, and discount level
- Statistical testing to validate observed differences in profitability across groups
- Simple modeling and statistical reasoning to support business interpretations

## Key Findings
- High sales volume does not necessarily translate to high profitability.
- Certain product categories and customer segments consistently underperform in profit.
- Higher discount levels are strongly associated with reduced or negative profit margins.
- Statistical tests support significant differences in profitability across segments,
  reinforcing insights observed in the exploratory analysis.

## Limitations
- The analysis is observational and does not imply causal relationships.
- External factors such as supplier costs, marketing strategy, and inventory constraints
  are not captured in the dataset.
- Results should be interpreted as descriptive insights rather than predictive outcomes.

## Project Structure

- notebooks/
  - 01_EDA_profitability_analysis.ipynb
  - 02_statistical_tests.ipynb
- data/
  - raw/
- README.md



## Tools Used
- Python
- pandas, NumPy
- matplotlib, seaborn
- SciPy / statsmodels (statistical analysis)

## Status
Complete. This project emphasizes exploratory analysis and statistical validation to
support business decision-making rather than predictive modeling.

