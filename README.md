# Direct Marketing Optimization Case Study

## Project Overview
This project focuses on maximizing revenue from direct marketing campaigns using machine learning models to predict customer propensity and optimize targeting strategies. The goal is to maximize revenue while adhering to constraints such as contact limitations and single-offer assignments.

## Repository Structure
```
├── data/
│   ├── DataScientist_CaseStudy_Dataset.csv               # The original RAW dataset
│   ├── X.csv                                             # Processed features dataset
│   ├── y.csv                                             # Processed target labels dataset
│
├── lib/
│   ├── train_functions.py                                # Functions for model training and evaluation
│   ├── optim_utils.py                                    # Reusable functions used in the Optimization step
│   ├── optim_functions.py                                # Objective function used in the Optimization step
│
├── output/
│   ├── top_15_pct_by_revenue.csv                         # CSV containing the final top 15% marketing recommendations
│   ├── top_prop_*.csv                                    # CSV containing the final predictions of the 3 propensity models trained
│
├── train_params/
│   ├── *.json                                            # JSON files of the corresponding best hyperparameters generated
│
├── part-1-eda.ipynb                                      # Exploratory Data Analysis (EDA), sanitation, and preprocessing
├── part-2-data-modelling.ipynb                           # Model training, benchmarking, and revenue optimization
├── executive-summary.pdf                                 # 2-page executive summary
├── requirements.txt                                      # List of dependencies
├── README.md                                             # Project documentation
```

## Methodology
### 1. Exploratory Data Analysis (EDA) & Preprocessing
- Conducted data cleaning and preprocessing.
- Handled missing values, outliers, redundant features, feature engineering, etc..
- Generated `X.csv` and `y.csv` as inputs for model training.

### 2. Model Training & Evaluation
- Built propensity models for:
  - **Consumer Loan**
  - **Credit Card**
  - **Mutual Fund**
- Used LGBM as the primary model as I believe it works well for imbalanced datasets.
- Used Optuna for hyperparameter tuning to optimize model performance.
- Benchmarked 3 different model configurations to select the best-performing ones.

### 3. Targeting Strategy & Revenue Optimization
- Applied the best-performing models to estimate customer likelihood of conversion.
- Selected the top 15% of customers that maximizes revenue.
- Produced the final targeting list as `top_15_pct_by_revenue.csv`.

## How to Use This Repository
### Prerequisites
Ensure you have Python installed. Clone this repository and install dependencies:
```sh
pip install -r requirements.txt
```

### Running the Notebooks
1. **Data Exploration & Preprocessing**:
   - Run `part-1-eda.ipynb` to clean and preprocess data.
   - Outputs: `X.csv` and `y.csv` in the `data/` folder.

2. **Model Training & Evaluation**:
   - Run `part-2-data-modelling.ipynb` to train models and generate predictions.
   - Outputs:
     - Final targeting strategy in `output/top_15_pct_by_revenue.csv`

### Reproducing Best Results
- The best hyperparameters are stored in `train_params/*.json`.
- To retrain using these parameters, load them in `part-2-data-modelling.ipynb` using the `load_params()` function in `lib.optim_utils`.
```python
from lib.optim_utils import load_params
```

## Contact
For any questions or clarifications, feel free to reach out:
- **Email**: reinbugnot@gmail.com
- **LinkedIn**: [Rein Bugnot](https://www.linkedin.com/in/reinbugnot/)
