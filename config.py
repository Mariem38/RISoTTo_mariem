# import sys
from datetime import datetime


config_data = {
    'dataset_filepath': "datasets/pdb_bprna_structures_nuc_v9_split.h5",  # PSU, 5MC, OMG; large entries split per chain / cropped
    # 90/5/5 split by MMseqs2 sequence cluster (80% id), PSU balanced (make_splits.py)
    'train_selection_filepath': "datasets/splits_mods_train.txt",
    'validation_selection_filepath': "datasets/splits_mods_validation.txt",
    'test_selection_filepath': "datasets/splits_mods_test.txt",
    'max_ba': 1,
    'max_size': 1024*8,  # 1024*12 runs out of memory on a 32 GB V100
    'min_num_res': 10,
    'r_noise': 0,
    'virt_cb': False,
    'partial': True,
}
#changed input and output layer size: N0 = 30 element features + 7 RNA classes (G, A, U, C, PSU, 5MC, OMG)
config_model = {
    "em": {'N0': 37, 'N1': 64},
    "sum": sum([
        [{'Ns': 64, 'Nh': 2, 'Nk':3, 'nn': 8}]*5,
        [{'Ns': 64, 'Nh': 2, 'Nk':3, 'nn': 16}]*5,
        [{'Ns': 64, 'Nh': 2, 'Nk':3, 'nn': 32}]*5,
        [{'Ns': 64, 'Nh': 2, 'Nk':3, 'nn': 64}]*5,
    ], []),
    "spl": {'N0': 64, 'N1': 64, 'Nh': 4},
    "dm": {'N0': 64, 'N1': 64, 'N2': 7},
}

# define run name tag
tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 'rna_v45_benchmark_grnade_das_bpRNA_cg_d5_ss64_partial'+tag, # P, C4', O2', N9/N1
    'output_dir': 'save',
    'reload': False,
    'device': 'cuda',
    'num_epochs': 100,
    'log_step': 512,
    'eval_step': 512*8,
    'eval_size': 512,
    'learning_rate': 1e-4,
    'pos_weight_factor': 0.9,
    'comment': "",
}
