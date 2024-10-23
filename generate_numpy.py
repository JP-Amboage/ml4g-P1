import pandas as pd
import numpy as np
import pyBigWig
from tqdm import tqdm
import click

BIN_SIZE = 64*2
SEQ_LENGTH = 32768//2

PATH_TRAIN_X1_FEATURES = 'data/ML4G_Project_1_Data/CAGE-train/X1_train_info.tsv'
PATH_TRAIN_X1_TARGETS = 'data/ML4G_Project_1_Data/CAGE-train/X1_train_y.tsv'

PATH_VAL_X1_FEATURES = 'data/ML4G_Project_1_Data/CAGE-train/X1_val_info.tsv'
PATH_VAL_X1_TARGETS = 'data/ML4G_Project_1_Data/CAGE-train/X1_val_y.tsv'


PATH_TRAIN_X2_FEATURES = 'data/ML4G_Project_1_Data/CAGE-train/X2_train_info.tsv'
PATH_TRAIN_X2_TARGETS = 'data/ML4G_Project_1_Data/CAGE-train/X2_train_y.tsv'

PATH_VAL_X2_FEATURES = 'data/ML4G_Project_1_Data/CAGE-train/X2_val_info.tsv'
PATH_VAL_X2_TARGETS = 'data/ML4G_Project_1_Data/CAGE-train/X2_val_y.tsv'


PATH_H3K4me1_X1_BW = 'data/ML4G_Project_1_Data/H3K4me1-bigwig/X1.bigwig'
PATH_H3K4me3_X1_BW = 'data/ML4G_Project_1_Data/H3K4me3-bigwig/X1.bw'
PATH_H3K9me3_X1_BW = 'data/ML4G_Project_1_Data/H3K9me3-bigwig/X1.bw'
PATH_H3K27ac_X1_BW = 'data/ML4G_Project_1_Data/H3K27ac-bigwig/X1.bigwig'
PATH_H3K27me3_X1_BW = 'data/ML4G_Project_1_Data/H3K27me3-bigwig/X1.bw'
PATH_H3K36me3_X1_BW = 'data/ML4G_Project_1_Data/H3K36me3-bigwig/X1.bw'

PATH_H3K4me1_X2_BW = 'data/ML4G_Project_1_Data/H3K4me1-bigwig/X2.bw'
PATH_H3K4me3_X2_BW = 'data/ML4G_Project_1_Data/H3K4me3-bigwig/X2.bw'
PATH_H3K9me3_X2_BW = 'data/ML4G_Project_1_Data/H3K9me3-bigwig/X2.bw'
PATH_H3K27ac_X2_BW = 'data/ML4G_Project_1_Data/H3K27ac-bigwig/X2.bw'
PATH_H3K27me3_X2_BW = 'data/ML4G_Project_1_Data/H3K27me3-bigwig/X2.bw'
PATH_H3K36me3_X2_BW = 'data/ML4G_Project_1_Data/H3K36me3-bigwig/X2.bw'


def build_features(chrom: str, TSS_start: int, strand: str, bw) -> np.ndarray:
    features = np.zeros(SEQ_LENGTH//BIN_SIZE)
    for i in range(features.shape[0]):
        start = TSS_start - SEQ_LENGTH//2 + i*BIN_SIZE
        end = start + BIN_SIZE
        values = bw.values(chrom, start, end)
        features[i] = np.mean(values)
    if strand == '-':
        features = features[::-1]
    return features


def build_numpy(features_path: str, targets_path: str, bws: list[str]) -> tuple[np.ndarray, np.ndarray]:

    df_features = pd.read_csv(features_path, delimiter='\t')

    X = np.zeros((df_features.shape[0], len(bws), SEQ_LENGTH//BIN_SIZE, ))
    y = np.zeros((df_features.shape[0], 1))

    if targets_path is not None:
        df_targets = pd.read_csv(targets_path, delimiter='\t')
        assert (df_features['gene_name'] == df_targets['gene_name']).all()
        y = df_targets['gex'].values

    opened_bws = [pyBigWig.open(bw) for bw in bws]

    for i in tqdm(range(df_features.shape[0])):
        row = df_features.iloc[i]
        chrom = row['chr']
        TSS_start = row['TSS_start']
        strand = row['strand']
        for j, bw in enumerate(opened_bws):
            X[i, j] = build_features(chrom, TSS_start, strand, bw)
    
    return X, y

def arg2paths(arg: str)-> tuple[str, str, list[str]]:
    if arg == 'X1_train':
        features_path = PATH_TRAIN_X1_FEATURES
        targets_path = PATH_TRAIN_X1_TARGETS
        bws = [PATH_H3K4me1_X1_BW, 
               PATH_H3K4me3_X1_BW, 
               PATH_H3K9me3_X1_BW, 
               PATH_H3K27ac_X1_BW, 
               PATH_H3K27me3_X1_BW, 
               PATH_H3K36me3_X1_BW]

    elif arg == 'X1_val':
        features_path = PATH_VAL_X1_FEATURES
        targets_path = PATH_VAL_X1_TARGETS
        bws = [PATH_H3K4me1_X1_BW, 
               PATH_H3K4me3_X1_BW, 
               PATH_H3K9me3_X1_BW, 
               PATH_H3K27ac_X1_BW, 
               PATH_H3K27me3_X1_BW, 
               PATH_H3K36me3_X1_BW]
        
    elif arg == 'X2_train':
        features_path = PATH_TRAIN_X2_FEATURES
        targets_path = PATH_TRAIN_X2_TARGETS
        bws = [PATH_H3K4me1_X2_BW, 
               PATH_H3K4me3_X2_BW, 
               PATH_H3K9me3_X2_BW, 
               PATH_H3K27ac_X2_BW, 
               PATH_H3K27me3_X2_BW, 
               PATH_H3K36me3_X2_BW]
        
    elif arg == 'X2_val':
        features_path = PATH_VAL_X2_FEATURES 
        targets_path = PATH_VAL_X2_TARGETS
        bws = [PATH_H3K4me1_X2_BW, 
               PATH_H3K4me3_X2_BW, 
               PATH_H3K9me3_X2_BW, 
               PATH_H3K27ac_X2_BW, 
               PATH_H3K27me3_X2_BW, 
               PATH_H3K36me3_X2_BW]
    
    return features_path, targets_path, bws

@click.command()
@click.argument('arg', type=str, required=True) #help='X1_train, X1_val, X2_train, X2_val'
def main(arg: str):
    
    print(f'-> Building numpy arrays for {arg}')
    
    features_path, targets_path, bws = arg2paths(arg)
    X1_train, y1_train = build_numpy(features_path, targets_path, bws)

    x_path = f'data/numpy/{arg}_X.npy'
    print(f'-> Saving features to {x_path}')
    np.save(f'data/numpy/{arg}_X.npy', X1_train)

    y_path = f'data/numpy/{arg}_y.npy'
    print(f'-> Saving targets to {y_path}')
    np.save(f'data/numpy/{arg}_y.npy', y1_train)
    
if __name__ == '__main__':
    main()