import os

root = '/mnt/disk/lym/'
datasets_root = os.path.join(root, 'COD10K/')
cod_training_root = os.path.join(datasets_root, 'TrainDataset')
chameleon_path = os.path.join(datasets_root, 'TestDataset/CHAMELEON')
camo_path = os.path.join(datasets_root, 'TestDataset/CAMO')
cod10k_path = os.path.join(datasets_root, 'TestDataset/COD10K')
nc4k_path = os.path.join(datasets_root, 'TestDataset/NC4K')
pvtv2_checkpoint_dir = '/mnt/disk/lym/model_results/backbone_weights/'