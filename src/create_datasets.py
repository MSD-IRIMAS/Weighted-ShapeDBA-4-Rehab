from aeon.clustering.averaging import elastic_barycenter_average
import numpy as np
import heapq
from src.utils import normalize_skeletons, create_directory
from aeon.distances import dtw_pairwise_distance

# for reproducibility of folds
np.random.seed(seed=42)

def load_base_dataset(original_dir, res_dir, exercise=1, num_folds=5):
    
    # path to output directory
    out_dir = res_dir + '/ex' + str(exercise) + '/experiment_dataset/'
    create_directory(out_dir)
    
    # load skeleton sequences (N,T,J,3)
    sequences = np.load(original_dir + '/ex' + str(exercise) + '/Skeletons.npy')
    # normalize sequences
    n_sequences, min_X, max_X, min_Y, max_Y, min_Z, max_Z = normalize_skeletons(X=sequences)
    # reshape to 3D (N,T,Jx3)
    sequences_3D = sequences.reshape((sequences.shape[0],sequences.shape[1],sequences.shape[2]*sequences.shape[3]))
    # permute (N,Jx3,T)
    sequences_3D = np.transpose(sequences_3D, axes=[0,2,1])
    
    # load corresponding scores (N,1)
    scores = np.load(original_dir + '/ex' + str(exercise) + '/Scores.npy')
    # normalize scores
    scores = np.squeeze(scores/100.0)
    
    # load healthy-unhealthy labels
    hu_labels = np.load(original_dir + '/ex' + str(exercise) + '/HU.npy')
    # get indexes of healthy and unhealthy
    unhealthy_indexes = np.squeeze(np.argwhere(hu_labels==1))
    healthy_indexes = np.squeeze(np.argwhere(hu_labels==0))
    # randomly shuffle the index before creating folds
    np.random.shuffle(unhealthy_indexes)
    folds_indexes = np.array_split(unhealthy_indexes,num_folds)
    
    # get healthy sequences (always included in the training set)
    healthy_sequences = sequences_3D[healthy_indexes]
    healthy_scores = scores[healthy_indexes]
    
    for i in range(len(folds_indexes)):
        
        create_directory(out_dir + 'fold' + str(i))
        
        # test fold
        test_fold = folds_indexes[i]
        test_sequences = sequences_3D[test_fold]
        test_scores = scores[test_fold]
        
        # train fold
        train_fold = folds_indexes.copy()
        train_fold.pop(i)
        unhealthy_train_sequences = sequences_3D[np.concatenate(train_fold)]
        unhealthy_train_scores = scores[np.concatenate(train_fold)]
        # unhealthy train sequences and scores are concatenated with the healthy ones
        train_sequences = np.vstack([healthy_sequences,unhealthy_train_sequences])
        train_scores = np.concatenate([healthy_scores,unhealthy_train_scores])
        
        # saving
        np.save(out_dir + 'fold' + str(i) + '/x_train_base_fold' + str(i) + '.npy', train_sequences)
        np.save(out_dir + 'fold' + str(i) + '/y_train_base_fold' + str(i) + '.npy', train_scores)
        np.save(out_dir + 'fold' + str(i) + '/indexes_train_fold' + str(i) + '.npy', np.concatenate([healthy_indexes,np.concatenate(train_fold)]))
        np.save(out_dir + 'fold' + str(i) + '/x_test_fold' + str(i) + '.npy', test_sequences)
        np.save(out_dir + 'fold' + str(i) + '/y_test_fold' + str(i) + '.npy', test_scores)
        np.save(out_dir + 'fold' + str(i) + '/indexes_test_fold' + str(i) + '.npy', test_fold)
        
        
def compute_dtw_matrix(original_dir, res_dir, exercise=1):
    
    # path to output directory
    out_dir = res_dir + '/ex' + str(exercise) + '/dtw_matrix/'
    create_directory(out_dir)
    
    # load skeleton sequences (N,T,J,3)
    sequences = np.load(original_dir + '/ex' + str(exercise) + '/Skeletons.npy')
    # normalize sequences
    n_sequences, min_X, max_X, min_Y, max_Y, min_Z, max_Z = normalize_skeletons(X=sequences)
    # reshape to 3D (N,T,Jx3)
    sequences_3D = sequences.reshape((sequences.shape[0],sequences.shape[1],sequences.shape[2]*sequences.shape[3]))
    # permute (N,Jx3,T)
    sequences_3D = np.transpose(sequences_3D, axes=[0,2,1])
    
    # compute DTW matrix from aeon
    dtw_matrix = dtw_pairwise_distance(sequences_3D)
    print(dtw_matrix.shape)
    
    # save matrix
    np.save(out_dir + 'dtw_matrix_ex' + str(exercise) + '.npy',dtw_matrix)


def extend_weighted_sdba(res_dir, exercise=1, num_folds=5, num_neighbors=1):
    
    # path to base directory
    base_dir = res_dir + '/ex' + str(exercise) + '/'
    
    # load dtw matrix and put diag values to np.inf
    dtw_matrix = np.load(base_dir + 'dtw_matrix/dtw_matrix_ex' + str(exercise) + '.npy')
    np.fill_diagonal(dtw_matrix, np.inf)
    
    # create lists for storing neighbor combinations and corresponding averages and scores
    # before computing average, we will check if it has been already done in previous folds
    list_of_all_neighbor_index = []
    list_of_all_average_sequences = []
    list_of_all_average_scores = []

    for fold in range(num_folds):
        print('fold ' + str(fold))
        # fold directory
        fold_dir = base_dir + 'experiment_dataset/fold' + str(fold) + '/'
        
        # load test indexes of the corresponding fold
        test_indexes = np.load(fold_dir + 'indexes_test_fold' + str(fold) + '.npy')
        
        # remove rows and cols corresponding to test indexes
        dtw_matrix_fold = np.delete(dtw_matrix, test_indexes, axis=0)
        dtw_matrix_fold = np.delete(dtw_matrix_fold, test_indexes, axis=1)
        
        # load train sequences and scores
        x_train_base = np.load(fold_dir + 'x_train_base_fold' + str(fold) + '.npy')
        y_train_base = np.load(fold_dir + 'y_train_base_fold' + str(fold) + '.npy')
        

        print('Neighbors: ' + str(num_neighbors))
        #prepare corresponding synthetic arrays of the same shape as base ones
        x_train_synthetic = np.zeros(x_train_base.shape)
        y_train_synthetic = np.zeros(y_train_base.shape)
    
        # loop over train indexes and sequences
        for i in range(x_train_base.shape[0]):
            
            # get base sequence and put in the array to average
            x_base = x_train_base[i]
            x_to_average = x_train_base[i:i+1]
            scores_to_average = y_train_base[i:i+1]
        
            # get k smalles distances and associated indexes
            nearest_index_min_dist = heapq.nsmallest(num_neighbors, enumerate(dtw_matrix_fold[i,:]), key=lambda x: x[1])
            nearest_indexes = [nim[0] for nim in nearest_index_min_dist]
            nearest_distances = [nim[1] for nim in nearest_index_min_dist]
            
            index_neighbors = [i] + nearest_indexes

            # check if same combination has already been computed in previous folds (to avoir repeating average computation)
            if fold>0:
                try:
                    # check if the combination is in list (meaning average is already computed)
                    ind = list_of_all_neighbor_index.index(index_neighbors)
                    # if yes get the stored average sequence and score
                    x_train_synthetic[i] = list_of_all_average_sequences[ind]
                    y_train_synthetic[i] = list_of_all_average_scores[ind]
                    
                #if the combination is not in the list to the average computation
                except:
                    # build array of sequences to average including the base sequence and the k nearests
                    x_to_average = np.concatenate([x_to_average,x_train_base[nearest_indexes]])
                    # compute weights according to formula and insert weight 1 in first position
                    weights = [1.0] + [np.exp(np.log(0.5)*(nd/nearest_distances[0])) for nd in nearest_distances]
                    # compute sdba and add the weighted average in synthetic array
                    x_train_synthetic[i] = elastic_barycenter_average(X=x_to_average,weights=weights,init_barycenter=x_to_average[0],distance='shape_dtw',)
                    
                    # build array of score to average including the base sequence and the k nearests
                    scores_to_average = np.concatenate([scores_to_average,y_train_base[nearest_indexes]])
                    # normalize wights before computing average score
                    weights_normalized = weights/np.sum(weights)
                    # compute the weighted average score and add it to the synthetic array
                    y_train_synthetic[i] = np.sum(weights_normalized*scores_to_average)
                    
                    # store resulting index combination, average sequence and score
                    list_of_all_neighbor_index.append(index_neighbors)
                    list_of_all_average_sequences.append(x_train_synthetic[i])
                    list_of_all_average_scores.append(y_train_synthetic[i])
            else:
                # if fold i 0 then all averages need to be computed
                
                # build array of sequences to average including the base sequence and the k nearests
                x_to_average = np.concatenate([x_to_average,x_train_base[nearest_indexes]])
                # compute weights according to formula and insert weight 1 in first position
                weights = [1.0] + [np.exp(np.log(0.5)*(nd/nearest_distances[0])) for nd in nearest_distances]
                # compute sdba and add the weighted average in synthetic array
                x_train_synthetic[i] = elastic_barycenter_average(X=x_to_average,weights=weights,init_barycenter=x_to_average[0],distance='shape_dtw')
                
                # build array of score to average including the base sequence and the k nearests
                scores_to_average = np.concatenate([scores_to_average,y_train_base[nearest_indexes]])
                # normalize wights before computing average score
                weights_normalized = weights/np.sum(weights)
                # compute the weighted average score and add it to the synthetic array
                y_train_synthetic[i] = np.sum(weights_normalized*scores_to_average)
                
                # store resulting index combination, average sequence and score
                list_of_all_neighbor_index.append(index_neighbors)
                list_of_all_average_sequences.append(x_train_synthetic[i])
                list_of_all_average_scores.append(y_train_synthetic[i])
            
        # save the extended arrays
        np.save(fold_dir + 'x_train_wsdba_fold' + str(fold) + '_nn' + str(num_neighbors) + '.npy', x_train_synthetic)
        np.save(fold_dir + 'y_train_wsdba_fold' + str(fold) + '_nn' + str(num_neighbors) + '.npy', y_train_synthetic)


def extend_noise(res_dir, exercise=1, num_folds=5):         
        
    # path to base directory
    base_dir = res_dir + '/ex' + str(exercise) + '/'
    
    for fold in range(num_folds):
        print('fold ' + str(fold))
        # fold directory
        fold_dir = base_dir + 'experiment_dataset/fold' + str(fold) + '/'
        
        # load test indexes of the corresponding fold
        test_indexes = np.load(fold_dir + 'indexes_test_fold' + str(fold) + '.npy')
        
        # load train sequences and scores
        x_train_base = np.load(fold_dir + 'x_train_base_fold' + str(fold) + '.npy')
        y_train_base = np.load(fold_dir + 'y_train_base_fold' + str(fold) + '.npy')
        
        noise = np.random.normal(0,0.1,x_train_base.shape)
        x_train_noisy = x_train_base + noise
        
        # save the extended arrays
        np.save(fold_dir + 'x_train_noisy_fold' + str(fold) + '.npy', x_train_noisy)
        np.save(fold_dir + 'y_train_noisy_fold' + str(fold) + '.npy', y_train_base)
        