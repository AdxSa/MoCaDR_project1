# MoCaDR_project1

## Movie Recommender System

This repository implements a simple movie‐recommendation system. Given a ratings dataset (`userId, movieId, rating`) in CSV format, we train and evaluate four matrix-factorization methods:

- **NMF** (Non-negative Matrix Factorization)  
- **SVD1** (Truncated SVD )  
- **SVD2** (Iterated SVD: “SVD2”)  
- **SGD** (Stochastic Gradient Descent on low-rank factors)  

All methods are evaluated via \(k\)-fold cross-validation to select the best rank (and regularization) before producing final predictions.

---

### Requirements  
The following libraries are required to run the code:  
- `numpy`   
- `pandas`
- `Sci-kit learn`
- `torch`  
  
These libraries can be installed using:  
```bash
pip install -r requirements.txt
```


## Repository structure
- **`report.pdf`** – Report which sums up whole project
- **`README.MD`** – This file
- **`project1_s340146_s336942`** – Actual project
- **`requirements.txt`** – list of needed libraries
- **`sample_test.csv`** - sample test from sample_project
- **`sample_test_with_ratings.csv`** - sample test with ratings from sample_project
- **`plots`** - Directory storing generated plots used in the report
- **`tools`** - tools directory from sample_project

### project structure
- **project1_s340146_s336942/**  
  - **data/** - data store directory
    - `ratings.csv`  
  - **modules/** - most of the project code, all important functions used in project
    - `__init__.py`
    - `helper_functions.py`
    - `impute_functions.py`
    - `plot_functions.py`   
    - `predict_functions.py`  
    - `train_functions.py`  
  - **models_trained/** - directory with stored all trained models 
  - **results/** - directory to store predictions
  - `main.py`  
---


## Instructions


### Examples
#### 1. **Training**
  Use the following promt to train SVD2 model on `ratings.csv` file (with automatically choosen best r hyperparameter) and save it as SVD2_model.pkl in /models_trained directory:
```
  python main.py --train yes --train_file project1_s340146_s336942/data/ratings.csv --model_path /models_trained/SVD2_model.pkl --alg SVD2
```


#### 2. **Predicting**
   Use the following promt to predict ratings with previously trained SVD2 model and save the predictions as SVD2_preds.csv in /results
```
  python main.py --predict yes --test_file sample_test.csv --model_path /models_trained/SVD2_model.pkl --output_file project1_s340146_s336942/results/SVD2_preds.csv
``` 

#### 3. **Evaluation**
```
  python tools/evaluate_solution.py --true_file sample_test_with_ratings.csv --pred_file project1_s340146_s336942/results/pred_name.csv     
```
  you can use different files, to get better results.

### All possible parser arguments
| Argument                 | Type   | Default                             | Description                                                                                   |
|--------------------------|--------|-------------------------------------|-----------------------------------------------------------------------------------------------|
| `--train`                | str    | `"no"`                              | Whether to run training. Use `"yes"` to train, `"no"` to skip.                                |
| `--predict`              | str    | `"no"`                              | Whether to run prediction. Use `"yes"` to predict, `"no"` to skip.                            |
| `--train_file`           | str    | `"data/ratings.csv"`                | Path to your training CSV (`userId,movieId,rating`).                                          |
| `--input_file`           | str    | `"data/preds.csv"`                  | Path to your test CSV for prediction (`userId,movieId`).                                      |
| `--model_path`           | str    | `"models_trained/ALL_models.pkl"`   | Where to save (in train mode) or load (in predict mode) your pickled model data.              |
| `--output_file`          | str    | `"predictions/preds.csv"`           | Where to write your predicted ratings (`userId,movieId,rating`).                              |
| `--alg`                  | str    | `"ALL"`                             | Which algorithm to use: one of `NMF`, `SVD1`, `SVD2`, `SGD` or `ALL`.                         |
|--------------------------|--------|-------------------------------------|-----------------------------------------------------------------------------------------------|
| `--r`                    | int    | 0                                   | r to use for training (0 -> searching for best r)                                             |
| `--print_rmse_plots`     | str    | "no"                                | RMSE plot generation: 'yes' while training to save used algorithm RMSE plot                   |
| `--print_impute_plots`   | str    | "no"                                | Imputation method comparison plot generation: 'yes' while training to save used algorithm plot|


