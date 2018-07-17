# Databricks notebook source
#download  from kaggle website
import os
dataset_path = os.path.join("/dbfs/FileStore/tables", 'all.zip')


# COMMAND ----------

#import zipfile

#with zipfile.ZipFile(dataset_path, "r") as z:
#    z.extractall("/dbfs/tmp/data")

# COMMAND ----------

#Load the order training set.
order_product_train_file = os.path.join("/tmp/data", 'order_products__train.csv')

order_product_train_raw = sc.textFile(order_product_train_file)
order_product_train_raw_header = order_product_train_raw.take(1)[0]

order_product_train = order_product_train_raw.filter(lambda line: line!=order_product_train_raw_header)\
    .map(lambda line: line.split(",")).map(lambda x: (x[1] + "_" + x[2], x[3])).groupByKey().cache()

order_product_train.take(3)


# COMMAND ----------

#Get count of  user and product combination
def get_counts(user_product_tuple):
    count = len(user_product_tuple[1])
    return  user_product_tuple[0].split('_')[0], user_product_tuple[0].split('_')[1],  count

userProductCountRDD = order_product_train.map(get_counts)
userProductCountRDD.take(3)

# COMMAND ----------

#Determining ALS parameter using small datasets
training_RDD, validation_RDD, test_RDD = userProductCountRDD.randomSplit([6, 2, 2], seed=0)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

# COMMAND ----------

#Running ALS from MLLib
from pyspark.mllib.recommendation import ALS
import math

seed = 5
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print ('For rank ' + str(rank) + 'the RMSE is ' + str(error))
    
    if error < min_error:
        min_error = error
        best_rank = rank

print ("The best model was trained with rank %s" + str(best_rank))

# COMMAND ----------

#Testing with the best model
model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    
print ('For testing data the RMSE is %s' + str(error))

# COMMAND ----------

#Load the orders from prior set
order_product_prior_file = os.path.join("/tmp/data", 'order_products__prior.csv')

order_product_prior_raw = sc.textFile(order_product_prior_file)
order_product_prior_raw_header = order_product_prior_raw.take(1)[0]

order_product_prior = order_product_prior_raw.filter(lambda line: line!=order_product_prior_raw_header)\
    .map(lambda line: line.split(",")).map(lambda x: (x[1] + "_" + x[2], x[3])).groupByKey().cache()

order_product_prior.take(3)



# COMMAND ----------

#Get count of  user and product combination

userProductCountPriorRDD = order_product_prior.map(get_counts)
userProductCountPriorRDD.take(3)

# COMMAND ----------

#Determining ALS parameter using small datasets
training_RDD, validation_RDD, test_RDD = userProductCountPriorRDD.randomSplit([6, 2, 2], seed=0)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

# COMMAND ----------

#Running ALS from MLLib
from pyspark.mllib.recommendation import ALS
import math

seed = 5
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print ('For rank ' + str(rank) + 'the RMSE is ' + str(error))
    if error < min_error:
        min_error = error
        best_rank = rank

print ("The best model was trained with rank %s" + str(best_rank))

# COMMAND ----------

#Testing with the best model
model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    
print ('For testing data the RMSE is %s' +  str(error))
