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
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

#logging.basicConfig(level=logging.DEBUG)

def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["image"], d["pose"], d["landmark"]

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
    db_name = args.db_name
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    validation_split = args.validation_split
    model_type = args.model_type
    image_size = 64

    logging.debug("Loading data...")

    if db_name == '300W_LP':
        db_list = ['AFW.npz','AFW_Flip.npz','HELEN.npz','HELEN_Flip.npz','IBUG.npz','IBUG_Flip.npz','LFPW.npz','LFPW_Flip.npz']
        image = []
        pose = []
        landmark = []
        for i in range(0,len(db_list)):
            image_temp, pose_temp, landmark_temp = load_data_npz('../data/type1/'+db_list[i])
            image.append(image_temp)
            pose.append(pose_temp)
            landmark.append(landmark_temp)

        image = np.concatenate(image,0)
        pose = np.concatenate(pose,0)
        landmark = np.concatenate(landmark, 0)

        # we only care the angle between [-99,99] and filter other angles
        x_data = []
        y_data = []
        z_data = []
        print(image.shape)
        print(pose.shape)
        for i in range(0,pose.shape[0]):
            temp_pose = pose[i,:]
            if np.max(temp_pose)<=99.0 and np.min(temp_pose)>=-99.0:
                x_data.append(image[i,:,:,:])
                y_data.append(pose[i,:].reshape(3))
                z_data.append(landmark[i,:].reshape(68 * 2))

        x_data = np.array(x_data)
        y_data = np.array(y_data)
        z_data = np.array(z_data)
        print(x_data.shape)
        print(y_data.shape)
        print(z_data.shape)
    elif db_name == 'synhead_noBIWI':
        print('[*] Not available')
        exit(0)
    elif db_name == 'BIWI':
        print('[*] Not available')
        exit(0)
    else:
        print('db_name is wrong!!!')
        return

    start_decay_epoch = [30,60]

    optMethod = Adam()

    stage_num = [3,3,3]
    lambda_d = 1
    num_classes = [3, 136] #3
    isFine = False

    if model_type == 0:
        model = SSR_net_ori_MT(image_size, num_classes, stage_num, lambda_d)()
        save_name = 'ssrnet_ori_mt'

    elif model_type == 1:
        model = SSR_net_MT(image_size, num_classes, stage_num, lambda_d)()
        save_name = 'ssrnet_mt'

    elif model_type == 2:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_capsule'+str_S_set
    
    elif model_type == 3:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_var_capsule'+str_S_set

    elif model_type == 4:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 8*8*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_noS_capsule'+str_S_set

    elif model_type == 5:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_NetVLAD(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_netvlad'+str_S_set

    elif model_type == 6:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Var_NetVLAD(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_var_netvlad'+str_S_set
    
    elif model_type == 7:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 8*8*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_noS_NetVLAD(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_noS_netvlad'+str_S_set

    elif model_type == 8:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Metric(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_metric'+str_S_set

    elif model_type == 9:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Var_Metric(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_var_metric'+str_S_set
    elif model_type == 10:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 8*8*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_noS_Metric(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_noS_metric'+str_S_set

    model.compile(optimizer=optMethod, loss=["mae", "mse"],loss_weights=[1, 2])

    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    logging.debug("Saving model...")
    mk_dir(db_name+"_models")
    mk_dir(db_name+"_models/"+save_name)
    mk_dir(db_name+"_checkpoints")
    plot_model(model, to_file=db_name+"_models/"+save_name+"/"+save_name+".png")
    for i_L,layer in enumerate(model.layers):
        if i_L >0 and i_L< len(model.layers)-1:
            if 'pred' not in layer.name and 'caps' != layer.name and 'merge' not in layer.name and 'model' in layer.name:
                plot_model(layer, to_file=db_name+"_models/"+save_name+"/"+layer.name+".png")
    

    decaylearningrate = TYY_callbacks.DecayLearningRate(start_decay_epoch)

    callbacks = [ModelCheckpoint(db_name+"_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto"), decaylearningrate
                        ]

    logging.debug("Running training...")
    
    # NO BIWI!!!!
    if db_name != 'BIWI':
        data_num = len(x_data)
        indexes = np.arange(data_num)
        np.random.shuffle(indexes)
        x_data = x_data[indexes]
        y_data = y_data[indexes]
        z_data = z_data[indexes]
        train_num = int(data_num * (1 - validation_split))
        
        x_train = x_data[:train_num]
        x_test = x_data[train_num:]
        y_train = [y_data[:train_num], z_data[:train_num]]
        y_test = [y_data[train_num:], z_data[train_num:]]
    elif db_name == 'BIWI':
        print('[*] Not available')
        exit(0)

        #train_num = np.shape(x_train)[0]

    hist = model.fit_generator(generator=data_generator_pose_landmark(X=x_train, Y=y_train, batch_size=batch_size),
                                       steps_per_epoch=train_num // batch_size,
                                       validation_data=(x_test, y_test),
                                       epochs=nb_epochs, verbose=1,
                                       callbacks=callbacks)
    
    logging.debug("Saving weights...")
    model.save_weights(os.path.join(db_name+"_models/"+save_name, save_name+'.h5'), overwrite=True)
    pd.DataFrame(hist.history).to_hdf(os.path.join(db_name+"_models/"+save_name, 'history_'+save_name+'.h5'), "history")


if __name__ == '__main__':
    main()