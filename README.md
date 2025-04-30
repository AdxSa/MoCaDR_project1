# MoCaDR_project1

1. **Training**
  Use the following command to train your model (with automatically choosen best r hyperparameter) and save it as model_name.pkl in /models subfolder:
```
  python main.py --train yes --train_file project1_s340146_s336942/data/ratings.csv --model_path models/model_name.pkl --alg nmf
```
  use svd1, svd2 instead nmf for svd1, svd2

2. **Predicting**
   Use the following to predict ratings with previously trained model in model_name.pkl file and save predictions at pred_name.csv at results
```
  python main.py --predict yes --test_file sample_test.csv --model_path models/model_name.pkl --output_file project1_s340146_s336942/results/pred_name.csv
```  
  you don't need to precise which decomposition method you used. Prediction function is one for all methods.

3. **Evaluation**
  Currently on file which is subset of training file (ratings.csv) so it's kinda stupid but you can check results anyway. Use command:
```
  python tools/evaluate_solution.py --true_file sample_test_with_ratings.csv --pred_file project1_s340146_s336942/results/pred_name.csv     
```
  you can use different files, to get better results.


# Movie Recommender System

This repository implements a simple movie‐recommendation system. Given a ratings dataset (`userId, movieId, rating`) in CSV format, we train and evaluate four matrix-factorization methods:

- **NMF** (Non-negative Matrix Factorization)  
- **SVD1** (Truncated SVD )  
- **SVD2** (Iterated SVD: “SVD2”)  
- **SGD** (Stochastic Gradient Descent on low-rank factors)  

All methods are evaluated via \(k\)-fold cross-validation to select the best rank (and regularization) before producing final predictions.

---

## Repository structure
- **`Report.pdf`** – Report which sums up whole project
- **`main.py`** – Main file parsing user's commands
- **`project1_s340146_s336942`** – Directory storing all the code and data used in the project
- **`requirements.txt`** – list of needed libraries  
- **`plots`** - Directory storing generated plots used in the report
- **`models`** - Directory storing trained models in .pkl format

---

## Requirements  
The following libraries are required to run the code:  
- `numpy`   
- `pandas`
- `Sci-kit learn`
- `torch`  
  
These libraries can be installed using:  
```bash
pip install -r requirements.txt
```



