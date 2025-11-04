## Decision-Trees-and-Random-Forest-Project

This project applies **Random Forest Classification** to predict whether a borrower will fully repay their loan using data from LendingClub.com.

Lending Club is a peer-to-peer lending platform that connects borrowers with investors.
As an investor, the goal is to identify borrowers with a **high probability of repayment** to minimize financial risk.

The dataset covers loans issued between **2007** and **2010**, before Lending Club became a public company.

## Dataset Description
The dataset contains borrower information such as credit score, income, loan purpose, and financial history.
Each record includes whether the loan was **fully paid back (1) or defaulted (0)**.

## Features
- credit.policy: 1 if the customer meets LendingClub’s credit underwriting criteria, 0 otherwise
- purpose: Purpose of the loan (credit_card, debt_consolidation, educational, major_purchase, small_business, all_other)
- int.rate: Interest rate (e.g., 0.11 for 11%) — higher rates indicate higher risk
- installment: Monthly payment owed by the borrower
- log.annual.inc: Logarithm of annual income
- dti: Debt-to-income ratio
- fico: FICO credit score
- days.with.cr.line: Number of days the borrower has had a credit line
- revol.bal: Revolving balance (unpaid credit card balance)
- revol.util: Revolving line utilization rate (percentage of credit line used)
- inq.last.6mths: Number of recent credit inquiries
- delinq.2yrs: Number of delinquent payments in the past 2 years
- pub.rec: Number of derogatory public records (bankruptcies, tax liens, etc.)

## Objectives
- Load and explore Lending Club loan data (2007–2010)
- Perform Exploratory Data Analysis (EDA) and visualize borrower distributions
- Train Decision Tree and Random Forest classifiers
- Compare model accuracy and interpret feature importance
- Predict whether borrowers will repay loans in full or default

## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- PyCharm

## How to Run
1. Clone the repository:  
```bash
git clone <your-repo-url>
```
2. Install dependencies:
 ```bash
pip install <name of libraries>
```
3. Run the main Python script:
 ```bash
python Decision Trees and Random Forest.py
```

## Notes
- The dataset used has been **cleaned of missing values (NA)** for convenience
- The project compares **Decision Tree** vs **Random Forest** performance
- Ideal for practicing **ensemble methods** and **feature importance analysis** in machine learning
- Based on real Lending Club data prior to its IPO in 2016
