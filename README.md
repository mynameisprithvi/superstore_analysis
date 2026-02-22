# Superstore Sales, Profitability, and Discount Analysis

## Objective
This project analyzes sales, profitability, and discount patterns in a retail superstore  
dataset to understand how pricing decisions, product structure, and customer segments  
influence business outcomes.

The work combines exploratory analysis, statistical validation, and predictive modeling  
to provide a structured view of how discounts relate to sales performance and profit.

---

## Key Questions
- How do sales and profit vary across product categories and customer segments?
- What is the relationship between discount levels and profitability?
- Are observed differences in profitability statistically significant?
- Do high sales volumes necessarily correspond to high profits?
- Can discounts be reasonably predicted from order and product information?

---

## Dataset
The dataset contains transactional-level retail data, including sales, profit, discount,  
product category, customer segment, and geographic information. The data is observational  
and reflects historical retail operations rather than controlled experiments.

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Examined distributions of sales, profit, and discount levels  
- Compared performance across product categories, customer segments, and regions  
- Identified patterns suggesting structural drivers of profit and loss  

### 2. Statistical Analysis
- Applied statistical tests to evaluate whether observed differences in profitability  
  across segments and categories were significant  
- Used statistical reasoning to support or challenge insights from the exploratory analysis  

### 3. Discount Prediction Modeling
- Built a supervised machine learning model to predict order-level discounts using  
  transactional, product, and regional features  
- Used a tree-based model to capture nonlinear and rule-based discount patterns  
- Performed sanity checks, baseline comparisons, and feature removal tests to assess  
  robustness and interpretation limits  
- Emphasized predictive performance rather than causal explanation  

---

## Key Findings
- High sales volume does not necessarily translate to high profitability.  
- Certain product categories and customer segments consistently underperform in profit.  
- Higher discount levels are strongly associated with reduced or negative profit margins.  
- Statistical tests confirm meaningful differences in profitability across segments.  
- Discounts can be predicted accurately from contextual order information, suggesting  
  consistent pricing patterns rather than random discounting.  

---

## Limitations
- The analysis is observational and does not imply causal relationships.  
- Some features may reflect underlying pricing or discount policies.  
- External factors such as supplier costs, marketing strategies, and inventory constraints  
  are not captured.  
- Predictive results should be interpreted as estimates of expected discounts under similar  
  conditions, not explanations of decision-making.  

---

## Project Structure

notebooks/
- 01_EDA_profitability_analysis.ipynb
- 02_statistical_tests.ipynb
- 03_discount_prediction_model.ipynb

data/
- raw/
- processed/

scripts/
- data_ingest.py
- preprocessing.py
- train.py

models/
- discount_model.joblib

.github/
- workflows/
  - train.yml

run_training.py
requirements.txt
README.md

---

## Tools Used
- Python  
- pandas, NumPy  
- matplotlib, seaborn  
- SciPy / statsmodels (statistical analysis)  
- scikit-learn, XGBoost (modeling)  

---

## Status
Complete.  
This project prioritizes clear analysis, statistical validation, practical prediction,  
and reproducible machine learning workflow design.

---

## Continuous Integration (CI)

This project includes an automated machine learning pipeline using GitHub Actions.

On every push to the main branch:
- Dependencies are installed in a clean environment  
- Raw data ingestion and preprocessing are executed  
- The training pipeline (`run_training.py`) runs end-to-end  
- The trained model artifact is generated and stored  

This ensures the full ML workflow (ingestion → preprocessing → training → artifact generation)  
is reproducible and environment-independent.
