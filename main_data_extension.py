import hydra
from omegaconf import DictConfig, OmegaConf

from src.create_datasets import *
from src.utils import create_directory

@hydra.main(config_name="config_hydra_extension.yaml", config_path="config")
def main(args: DictConfig):
    
    # save configuration file of the experiment used
    with open("config.yaml", "w") as f:
        OmegaConf.save(args, f)

    # create directory if not existing
    create_directory(args.res_dir)
    
    # load base dataset from original folder and create the folds
    load_base_dataset(original_dir=args.original_dir,res_dir=args.res_dir,exercise=args.exercise,num_folds=args.num_folds)
    
    # perform extension using a noisy version of base data (baseline)
    extend_noise(res_dir=args.res_dir,exercise=args.exercise,num_folds=args.num_folds)
    
    # compute DTW once per exercise to find nearest neighbors
    compute_dtw_matrix(original_dir=args.original_dir,res_dir=args.res_dir,exercise=args.exercise)
    # loop over neighbors
    for nn in args.num_neighbors_wsdba:
        # perform extension using weighted shape DBA
        extend_weighted_sdba(res_dir=args.res_dir,exercise=args.exercise, num_folds=args.num_folds,num_neighbors=nn)
        
        
if __name__ == "__main__":
    main()
