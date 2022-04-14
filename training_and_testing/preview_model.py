import os
import sys
sys.path.append('..')
import logging
import argparse
import pandas as pd
import numpy as np

from lib.FSANET_model import *
from lib.SSRNET_model import *

import TYY_callbacks
from TYY_generators import *

from keras.utils import np_utils
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

# logging.basicConfig(level=logging.DEBUG)

def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["image"], d["pose"]

def mk_dir(dir):
    try:
        os.mkdir( dir )
    except OSError:
        pass


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for head pose estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=90,
                        help="number of epochs")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio")
    parser.add_argument("--model_type", type=int, default=3,
                        help="type of model")
    parser.add_argument("--db_name", type=str, default='300W_LP',
                        help="type of model")

    args = parser.parse_args()
    return args



def main():
    args = get_args()
    model_type = args.model_type
    image_size = 64

    print("[*] Starting Preview")

    stage_num = [3,3,3]
    lambda_d = 1
    pose_classes = 3
    landm_classes = 68
    isFine = False

    if model_type == 0:
        model = SSR_net_ori_MT(image_size, pose_classes,landm_classes, stage_num, lambda_d)()
        save_name = 'ssrnet_ori_mt'

    elif model_type == 1:
        model = SSR_net_MT(image_size, pose_classes,landm_classes, stage_num, lambda_d)()
        save_name = 'ssrnet_mt'

    elif model_type == 2:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Capsule(image_size, pose_classes,landm_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_capsule'+str_S_set
    
    elif model_type == 3:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Var_Capsule(image_size, pose_classes,landm_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_var_capsule'+str_S_set

    elif model_type == 4:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 8*8*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_noS_Capsule(image_size, pose_classes,landm_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_noS_capsule'+str_S_set

    elif model_type == 5:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_NetVLAD(image_size, pose_classes,landm_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_netvlad'+str_S_set

    elif model_type == 6:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Var_NetVLAD(image_size, pose_classes,landm_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_var_netvlad'+str_S_set
    
    elif model_type == 7:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 8*8*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_noS_NetVLAD(image_size, pose_classes,landm_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_noS_netvlad'+str_S_set

    elif model_type == 8:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Metric(image_size, pose_classes,landm_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_metric'+str_S_set

    elif model_type == 9:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Var_Metric(image_size, pose_classes,landm_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_var_metric'+str_S_set
    elif model_type == 10:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 8*8*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_noS_Metric(image_size, pose_classes,landm_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_noS_metric'+str_S_set

    print("Model Previewer")
    print(model.summary())

if __name__ == '__main__':
    main()