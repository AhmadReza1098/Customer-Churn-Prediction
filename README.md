![Customer Churn Visualization](images/customer_churn_visualise.jpg)

# Lloyds Customer Churn Prediction

Building a "High-Recall Safety Net" to proactively identify at-risk bank customers before they leave.

---

## 📌 Table of Contents

- [Overview](#overview)  
- [Business Problem](#business-problem)  
- [Dataset](#dataset)  
- [Tools & Technologies](#tools--technologies)  
- [Project Structure](#project-structure)  
- [Data Cleaning & Preparation](#data-cleaning--preparation)  
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
- [Research Questions & Key Findings](#research-questions--key-findings)  
- [How to Run This Project](#how-to-run-this-project)  
- [Future Enhancements](#future-enhancements)  
- [Author & Contact](#author--contact)  

---

## Overview

In the world of banking, "silent churn" is a massive problem. Customers often disengage gradually logging in less, transacting less before they finally close their account. By the time they leave, it's often too late to win them back.

In this project, I didn't just build a model to predict churn; I built a **proactive safety net**. Instead of chasing "vanity metrics" like 99% accuracy (which often hides the truth in imbalanced data), I optimized this model for **Business Value**. My goal was to maximize **Recall** ensuring the bank catches as many at-risk customers as possible, even if it means raising a few false alarms.

---

## Business Problem

SmartBank, a subsidiary of Lloyds Banking Group, is facing increasing customer churn, particularly among young professionals and small business customers, who are switching to competitors offering more attractive digital experiences, pricing, and service quality. This rising churn is directly eroding customer lifetime value and revenue, while also increasing acquisition costs to replace lost customers.

- Which products and categories drive **maximum value**.  
- Where **discounts** and **stock levels** are not aligned (e.g., high price but out of stock).  
- How much **revenue is locked** in current inventory.  

**This project aims to:**

- Identify which customer segments (e.g. age groups, professions, small businesses) are at highest risk of churn, based on their demographics, product holdings, and engagement behaviour.
- Understand behavioural and financial drivers of churn, such as declining transaction activity, reduced product usage, frequent complaints, or lower digital engagement with SmartBank channels.
- Quantify the impact of service quality and interaction history (e.g. complaints, resolution status, call-centre or branch visits, response times) on churn propensity.

- Build a robust, data-driven churn prediction dataset that can later feed machine learning models to flag at-risk customers early, enabling timely retention actions.

---

## Dataset

- Source: Forage (LLOYDS Data Science & Analytics) 

Key columns (example):

- `Customer ID` – Unique identifier.
- `LoginFrequency` – How often the customer logs into the app/web.
- `Avg_Spend/ Total_Spend` – Transactional volume metrics.
- `Service_Calls` – Number of calls to customer support.  
- `MaritalStatus,Age,Gender` – Demographic features.
- `Churn` – Target variable (1 = Left, 0 = Stayed).   

---

## Tools & Technologies

* **Language:** Python 3.9+
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest, GridSearch, SMOTE), XGBoost
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / Google Colab


---

## Project Structure

```
Lloyds-Customer-Churn-Prediction/
│
├── README.md                          
├── requirements.txt                   
├── Task2_Model_Predictions_Final.csv  
│
├── notebooks/
│   ├── 01_Data_Cleaning_EDA.ipynb    
│   ├── 02_Model_Building_Tuning.ipynb 
│
└── reports/
    └── Executive_Summary.pdf          
```


## Data Cleaning & Preparation

To tackle the noisy and imbalanced nature of the dataset, I adopted a "High-Sensitivity Safety Net" strategy:  

- **1. Data Preprocessing:**
- Handled missing values and scaled numerical features.
- Applied **SMOTE** to balance the training data (creating synthetic examples of churners).
- ** 2. Model Selection:**
- Compared **Logistic Regression, XGBoost, and Random Forest**.
- Selected **Random Forest** because it offered superior stability and interpretability for this specific dataset.
- **3. Strategic Tuning (The Key Differentiator):** 
- Used `GridSearchCV` to optimize hyperparameters.
- Crucially, lowered the decision threshold to 0.25. This prioritized Recall over Precision, accepting more false alarms to ensure fewer missed churners.

These steps ensure all later EDA and insights use reliable inventory data.

---

## Exploratory Data Analysis (EDA)

Before building any models, I needed to understand the "shape" of the business problem. I didn't just look for missing values; I looked for the story behind the data.

- **The "Imbalance Trap":** The dataset was heavily skewed—the vast majority of customers were "Retained." This immediately flagged that standard accuracy metrics would be misleading (a model could just guess "Stay" for everyone and still get 85% accuracy). This finding drove my decision to use **SMOTE** (to balance the training data) and focus on **Recall**. 

- **Behavior > Demographics:** during the correlation analysis, I noticed that behavioral metrics (like `LoginFrequency` and `Avg_Spend`) showed stronger variances between churners and non-churners than static demographics like `Age` or `Gender`. 
 
- **The "High-Spender" Paradox:** I expected low-value customers to be the ones leaving. Instead, the distribution plots revealed that churn was prevalent among customers with *higher* average transaction values, indicating a risk of losing premium clients.

---

## Research Questions & Key Findings

Instead of just dumping model stats, I framed my analysis around three critical business questions:

- **Question 1: What is the single biggest "Red Flag" that a customer is about to leave?**
- **Findings:**  **Digital Silence.**
- The Random Forest feature importance analysis revealed that `LoginFrequency` is the #1 predictor of churn.
- **The Insight:** Customers don't just disappear overnight. They disengage digitally first—logging in less frequently—weeks before they actually close their account.
- ** Business Impact:** This "digital silence" is a massive opportunity for early intervention before the customer is gone for good.

- **Question 2: Are we losing our most valuable customers?**
- **Finding:** **Yes**, High-Value customers are highly volatile.
Both `Avg_Spend` and `Total_Spend` ranked in the top 3 drivers of churn.  
- **The Insight:** Counter-intuitively, the customers spending the most are often the most likely to leave. This suggests they are sensitive to service quality and likely have attractive offers from competitors.
- **Business Impact:** Retention budgets shouldn't be spread evenly; they must be aggressively targeted at this top 20% spending tier.

- **Question 3: Is it better to be "Accurate" or "Safe"?**
- **Finding:** **Safety wins.**
A standard model achieved decent accuracy but missed nearly 90% of the actual churners.
- **The Insight:** By optimizing for **Recall** and lowering the decision threshold to **0.25**, I built a "Safety Net" model.
- **Business Impact:** We accepted a lower overall accuracy (43%) to achieve a **64% detection rate** (Recall). In banking, catching a departing customer is worth the small cost of investigating a false alarm.

---

## How to Run This Project

1. **Clone the Repository**
   - Open your terminal and run:
2. **Install Dependencies**
   - Ensure you have Python 3.9+ installed.
   - Install the required libraries using pip:

3. **Launch the Notebook**
   - Open Jupyter Notebook or Jupyter Lab:
     ```bash
     jupyter notebook
     ```   
4. **Run the Analysis**
   - Run all cells in the notebook to:
     - Load and preprocess the data.
     - Train the Random Forest & XGBoost models.
     - Execute the GridSearch tuning.
     - Generate the final "Safety Net" confusion matrix and Recall scores.

> **Note:** I have set `random_state=42` throughout to ensure you get the exact same results (64% Recall) as in my report.

---

## Future Enhancements

- **Integrate Competitor Data:** Incorporate external market data (e.g., competitor interest rates) to understand the "why" behind unexplained churn.
- **Advanced NLP Analysis:** Apply **Sentiment Analysis** to customer service call logs to detect frustration levels before a customer even hangs up.
- **Real-Time Monitoring:** Build a **Power BI dashboard** that tracks `LoginFrequency` daily, enabling the retention team to act on "Digital Silence" alerts immediately.

---

## Author & Contact

**Ahmad Reza**  
Aspiring Data Analyst – SQL & BI  

- 📧 Email: ahmadreza6122@gmail.com  
- 🔗 LinkedIn: www.linkedin.com/in/ahmad-reza-econ  
- 🔗 https://github.com/AhmadReza1098  

Feel free to use or adapt this project as part of your analytics portfolio.





