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
