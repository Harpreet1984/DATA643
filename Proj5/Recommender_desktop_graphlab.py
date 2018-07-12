import matplotlib as mpl
#mpl.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import graphlab as gl
import time

start = time.time()

col_names = ["user_id", "item_id", "rating", "timestamp" ]
data = pd.read_table("C:\\Users\\naman\\Documents\\Harpreet\\CUNY\\Data_602\\Assignment-3\\Mitxpro\\u.data", names=col_names)
data.info()


##prepare for graphlib
sf=gl.SFrame(data)
print (sf)


#Creating Train, Validate, Test split
sf_train, sf_test = sf.random_split(.7)
sf_train, sf_validate = sf_train.random_split(.75)


#colaborative filtering, we initialize the model with training data and validation data to find best regularization term.
regularization_terms = [10**-5,10**-4,10**-3,10**-2,10**-1]
best_regularization_term=0
best_RMSE = np.inf
for regularization_term in regularization_terms:
    factorization_recommender = gl.recommender.factorization_recommender.create(sf_train, target='rating', regularization=regularization_term, solver= "als" )
    evaluation = factorization_recommender.evaluate_rmse(sf_validate, 'rating')
    if evaluation['rmse_overall'] < best_RMSE :
        best_RMSE = evaluation['rmse_overall']
        best_regularization_term = regularization_term
print ("Best Regularization Term " + str(best_regularization_term))
print ("Best Validation RMSE Achieved " +  str(best_RMSE))       

factorization_recommender = gl.recommender.factorization_recommender.create(sf_train, target= 'rating', regularization=best_regularization_term)
print ("Test RMSE on best Model " + str (factorization_recommender.evaluate_rmse(sf_test, 'rating')['rmse_overall']))
  
end = time.time()
print (end  - start)

#precision/recall matrix for each model.
models = [ factorization_recommender]
model_names = ['factorization']
precision_recall = gl.recommender.util.compare_models(sf_test, models, metric='precision_recall', model_names=model_names)

