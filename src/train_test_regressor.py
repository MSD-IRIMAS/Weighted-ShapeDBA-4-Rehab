import numpy as np
from sklearn.metrics import mean_absolute_error
from aeon.regression.deep_learning import FCNRegressor
import tensorflow as tf
import pandas as pd
import os

def create_and_train_regressor(res_dir, exercise, fold, init_name, best_name,x_train,y_train,nb_epochs,batch_size,regressor_archi='FCN'):
    
    # path to output directory for models
    out_dir_models = res_dir + '/ex' + str(exercise) + '/models/fold' + str(fold) + "/"
    
    if not os.path.exists(out_dir_models):
        os.makedirs(out_dir_models)

    # instanciate a FCN regressor for base data
    if regressor_archi=='FCN':
        regressor = FCNRegressor(save_best_model=True, best_file_name=best_name + '_fold_' + str(fold),
                                 save_init_model=True, init_file_name=init_name + '_fold_' + str(fold),
                                 file_path=out_dir_models,
                                 batch_size=batch_size, n_epochs=nb_epochs)
    else:
        print('Architecture not implemented (yet)')
        exit()

    regressor.fit(x_train, y_train)
            

def eval_regressor(res_dir, exercise, fold,best_name,x_test,y_test):
    
    # path to output directory for models
    out_dir_models = res_dir + '/ex' + str(exercise) + '/models/fold' + str(fold) + "/"

    # load best model
    best_file_name = out_dir_models + best_name + '_fold_' + str(fold)
    regressor_model = tf.keras.models.load_model(best_file_name + ".keras")
    
    # evaluate
    pred_scores = regressor_model.predict(x_test.transpose(0, 2, 1),verbose=False)
    mae = mean_absolute_error(y_test, pred_scores)
    
    return mae