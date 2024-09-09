import numpy as np
from sklearn.metrics import mean_absolute_error
from aeon.regression.deep_learning import FCNRegressor
import tensorflow as tf
import pandas as pd
import os

def create_and_train_regressor(res_dir, exercise, fold, init_name, best_name,x_train,y_train,nb_epochs,batch_size,regressor_archi='FCN'):
    
    # path to output directory for models
    out_dir_models = res_dir + '/ex' + str(exercise) + '/models/fold'

    # instanciate a FCN regressor for base data
    best_file_name = out_dir_models + str(fold) + '/' + best_name + '_fold_' + str(fold)
    init_file_name = out_dir_models + str(fold) + '/' + init_name + '_fold_' + str(fold) + '.hdf5'
    if regressor_archi=='FCN':
        regressor = FCNRegressor(n_epochs=1,save_best_model=True, best_file_name=best_file_name)
    else:
        print('Architecture not implemented (yet)')
        exit()
    regressor_model = regressor.build_model((x_train.shape[2],x_train.shape[1]))
    regressor_model.save(init_file_name)
            
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=50, min_lr=0.0001),
                    tf.keras.callbacks.ModelCheckpoint(filepath=best_file_name + ".hdf5",monitor="loss",save_best_only=True,),]
            
    regressor_model.fit(x_train.transpose(0, 2, 1),y_train,batch_size=batch_size,epochs=nb_epochs,verbose=False,callbacks=callbacks)
            

def eval_regressor(res_dir, exercise, fold,best_name,x_test,y_test):
    
    # path to output directory for models
    out_dir_models = res_dir + '/ex' + str(exercise) + '/models/fold'

    # load best model
    best_file_name = out_dir_models + str(fold) + '/' + best_name + '_fold_' + str(fold)
    regressor_model = tf.keras.models.load_model(best_file_name + ".hdf5")
    
    # evaluate
    pred_scores = regressor_model.predict(x_test.transpose(0, 2, 1),verbose=False)
    mae = mean_absolute_error(y_test, pred_scores)
    
    return mae