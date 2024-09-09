import hydra
from omegaconf import DictConfig, OmegaConf

from src.train_test_regressor import *

@hydra.main(config_name="config_hydra_regression.yaml", config_path="config")
def main(args: DictConfig):
    
    # path to skeleton sequence directory
    data_dir = args.res_dir + '/ex' + str(args.exercise) + '/experiment_dataset/fold'

    # path to output directory for metric file
    out_dir_mae = args.res_dir + '/ex' + str(args.exercise) + '/mae_errors/fold'

    for fold in range(args.num_folds):
        
        print('\tFold ' + str(fold))
        
        # array for storing mae errors
        maes_array = np.zeros((1,len(args.num_neighbors_wsdba)+2))
        
        # load test data
        x_test = np.load(data_dir + str(fold) + '/x_test_fold' + str(fold) + '.npy')
        y_test = np.load(data_dir + str(fold) + '/y_test_fold' + str(fold) + '.npy')
        
        
        
        # load original train data (base)
        x_train_base = np.load(data_dir + str(fold) + '/x_train_base_fold' + str(fold) + '.npy')
        y_train_base = np.load(data_dir + str(fold) + '/y_train_base_fold' + str(fold) + '.npy')
        
        # regression on base data
        create_and_train_regressor(res_dir=args.res_dir,exercise=args.exercise,fold=fold,
                                   init_name='init_model_base', best_name='best_model_base',
                                   x_train=x_train_base,y_train=y_train_base,
                                   nb_epochs=args.nb_epochs,batch_size=args.batch_size)
        
        # evaluate the trained regressor on test set
        mae_base = eval_regressor(res_dir=args.res_dir,exercise=args.exercise,fold=fold,
                            best_name='best_model_base',x_test=x_test,y_test=y_test)
        maes_array[0,0] = mae_base
        
        
        
        # load noisy train data
        x_train_noisy = np.load(data_dir + str(fold) + '/x_train_noisy_fold' + str(fold) + '.npy')
        y_train_noisy = np.load(data_dir + str(fold) + '/y_train_noisy_fold' + str(fold) + '.npy')
            
        # regression on noisy data
        create_and_train_regressor(res_dir=args.res_dir,exercise=args.exercise,fold=fold,
                                   init_name='init_model_noisy', best_name='best_model_noisy',
                                   x_train=x_train_noisy,y_train=y_train_noisy,
                                   nb_epochs=args.nb_epochs,batch_size=args.batch_size)
        
        # evaluate the trained regressor on test set
        mae_noisy = eval_regressor(res_dir=args.res_dir,exercise=args.exercise,fold=fold,
                            best_name='best_model_noisy',x_test=x_test,y_test=y_test)
        maes_array[0,1] = mae_noisy
        
        
        
        # on extended data using weighted sdba
        # loop over over various sets considering various neighbors
        for i,nn in enumerate(args.num_neighbors_wsdba):
            
            print('neighbor', i,nn)
            
            # load wsdba train data
            x_train_wsdba = np.load(data_dir + str(fold) + '/x_train_wsdba_fold' + str(fold) + '_nn' + str(nn+1) + '.npy')
            y_train_wsdba = np.load(data_dir + str(fold) + '/y_train_wsdba_fold' + str(fold) + '_nn' + str(nn+1) + '.npy')
                
            # regression on wsdba data
            create_and_train_regressor(res_dir=args.res_dir,exercise=args.exercise,fold=fold,
                                    init_name='init_model_wsdba_nn' + str(nn), best_name='best_model_wsdba_nn' + str(nn),
                                    x_train=x_train_wsdba,y_train=y_train_wsdba,
                                    nb_epochs=args.nb_epochs,batch_size=args.batch_size)
            
            # evaluate the trained regressor on test set
            mae_wsdba_nn = eval_regressor(res_dir=args.res_dir,exercise=args.exercise,fold=fold,
                                best_name='best_model_wsdba_nn' + str(nn),x_test=x_test,y_test=y_test)
            maes_array[0,i+2] = mae_wsdba_nn
            

if __name__ == "__main__":
    main()