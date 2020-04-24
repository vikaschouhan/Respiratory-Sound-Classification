import pandas as pd
import numpy as np
import librosa
import librosa.display
import sys
import math
import joblib
from   concurrent import futures
from   functools import partial
from   tqdm import tqdm
from   sklearn.model_selection import train_test_split
from   sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from   sklearn.preprocessing import LabelEncoder
import split_folders

from   datetime import datetime
import os
from   os import listdir
from   os.path import isfile, join
import shutil

from   tensorflow.keras.models import Sequential
from   tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from   tensorflow.keras.utils import to_categorical
from   tensorflow.keras.callbacks import ModelCheckpoint
from   tensorflow.keras import optimizers
from   tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import seaborn as sns

import tf2_models.tf2_models as tf2_models

###############################################
# Make directory
def mkdir(dir):
    if dir == None:
        return None
    # endif
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)
    # endif
# enddef

################################################
# Use multiprocessing to spawn worker processes to process
# huge amount of data parallely
# @args
#    fn_tor      => function to be parallelized
#    iterator    => a data of type iterator (list for eg.) on which
#                   fn_tor needs to be called for each element of the iterator
#                   and results collected.
#    n_jobs      => number of jobs to run in parallel
#    *args & **kwargs => extra arguments for fn_tor which are passed as such.
#
# @returns
#    list of all values when fn_tor is called on each element in iterator
#
# Help:
# This basically is same as 
# ret_list = []
# for item in iterator:
#     ret_list += fn_tor(item)
# endfor
def spawn(fn_tor, iterator, n_jobs=4, *args, **kwargs):
    exector  = futures.ProcessPoolExecutor(max_workers=n_jobs)
    lmbda_fn = partial(fn_tor, *args, **kwargs) #lambda x: fn_tor(x, *args, **kwargs)
    results  = exector.map(lmbda_fn, iterator)
    rresults = list(tqdm(results, total=len(iterator)))
    return rresults
# enddef

###########################################
# Initiate dataset hierarchy of respiratory sounds database
# @args
#     root_dir  => root directory of the dataset
def get_dataset_dirs(root_dir, verbose=True):
    v_dict                        = {}
    v_dict['dataset_root_dir']    = root_dir
    v_dict['demographic_file']    = '{}/demographic_info.txt'.format(root_dir)
    v_dict['audio_txt_pdir']      = '{}/respiratory_sound_database/Respiratory_Sound_Database'.format(root_dir)
    v_dict['audio_txt_dir']       = '{}/audio_and_txt_files'.format(v_dict['audio_txt_pdir'])
    v_dict['filename_diff_file']  = '{}/filename_differences.txt'.format(v_dict['audio_txt_pdir'])
    v_dict['filename_fmt_file']   = '{}/filename_format.txt'.format(v_dict['audio_txt_pdir'])
    v_dict['patient_diag_file']   = '{}/patient_diagnosis.csv'.format(v_dict['audio_txt_pdir'])
   
    if verbose:
        print('Dataset root_dir    = ', v_dict['dataset_root_dir'])
        print('demographic_file    = ', v_dict['demographic_file'])
        print('audio_txt_pdir      = ', v_dict['audio_txt_pdir'])
        print('audio_txt_dir       = ', v_dict['audio_txt_dir'])
        print('filename_diff_file  = ', v_dict['filename_diff_file'])
        print('filename_fmt_file   = ', v_dict['filename_fmt_file'])
        print('patient_diag_file   = ', v_dict['patient_diag_file'])
    # endif

    return v_dict
# enddef

###################################################
# Get list of all audio files
def get_filenames(paths, full_path=False):
    file_names = [f for f in listdir(paths['audio_txt_dir']) if (isfile(join(paths['audio_txt_dir'], f)) and f.endswith('.wav'))]
    if full_path:
        file_names = [join(paths['audio_txt_dir'], f) for f in file_names]
    # endif
    return file_names
# enddef

####################################################
# Get list of all files after stripping file extentions.
def get_filenames_without_extension(paths, full_path=False):
    file_names = list(set([os.path.splitext(f)[0] for f in listdir(paths['audio_txt_dir']) if (isfile(join(paths['audio_txt_dir'], f)))]))
    if full_path:
        file_names = [join(paths['audio_txt_dir'], f) for f in file_names]
    # endif
    return file_names
# enddef

####################################################
# Helper functions
def __get_x_files(file_names, dirname=None, x='wav'):
    suff = '.wav' if x == 'wav' else '.txt'
    if dirname:
        return [os.path.join(dirname, x) + suff for x in file_names]
    else:
        return [x + suff for x in file_names]
    # endif
# enddef

def get_wav_files(file_names, dirname=None):
    return __get_x_files(file_names, dirname, x='wav')
# enddef
def get_txt_files(file_names, dirname=None):
    return __get_x_files(file_names, dirname, x='txt')
# enddef

############################################################
# Get patient IDs correspond to file names
# We know from the dataset description that patient id is encoded
# in the file name itself.
def get_patient_ids(file_names):
    p_id_in_file = [] # patient IDs corresponding to each file
    for name in file_names:
        p_id_in_file.append(int(name[:3]))
    # endfor
    p_id_in_file = np.array(p_id_in_file) 
    return p_id_in_file
# enddef

#############################################################
# Feature extractors
# MFCC coefficients
def e_mfcc(audio, sr=22050, n_mfcc=40):
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
# enddef
# Normalized Mel spectrogram (between -1.0 & 1.0)
def e_mel_spectrogram(audio, sr=22050, n_fft=2048, hop_length=512, ref=40):
    assert ref in [40, 80], 'ref={} should be 40 or 80 only.'.format(ref)
    return (librosa.power_to_db(librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length), ref=np.max) + ref)/ref
# endif
# Plain Mel spectrogram
def e_mel_spectrogram2(audio, sr=22050, n_fft=2048, hop_length=512):
    return librosa.power_to_db(librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length), ref=np.max)
# enddef

#############################################################
# Read single audio file. Zero pad if necessary
def read_and_zero_pad(file_name, max_slen=87, sr=22050, verbose=True):
    data, sr = librosa.load(file_name, sr=sr, mono=True, duration=max_slen)
    obuf_len = len(data)
    nbuf_len = max_slen*sr
    try:
        data     = librosa.util.pad_center(data, max_slen*sr)
    except:
        print('WARNING:: Unable to read {}'.format(file_name))
        sys.exit(-1)
    # endtry

    if (obuf_len > nbuf_len) and verbose:
        print('WARNING:: Truncating {} of length {} to {}'.format(file_name, obuf_len, nbuf_len))
    # endif

    return data, sr
# enddef

##############################################################
# Read all audio files in the dataset. Wrapper over read_and_zero_pad()
# @args
#    paths        => paths dictionary returned from get_dataset_dirs()
#    max_slen     => max seconds to read from each audio file
#    sr           => sample rate to be used
#    file_names   => List of specific files to read. When this is passed, only
#                    files in this list are read.
#    n_jobs       => number of jobs to parallelize
# @returns
#    dictionary of audio samples hashed by file names
def read_all_files(paths, max_slen=87, sr=22050, verbose=True, file_names=None, n_jobs=4):
    fnames_we  = get_filenames_without_extension(paths)
    fnames_we  = list(filter(lambda x: x in fnames_we, file_names)) if file_names else fnames_we
    file_names = get_wav_files(fnames_we, dirname=paths['audio_txt_dir'])

    #files_data = [read_and_zero_pad(x, max_slen=max_slen, sr=sr, verbose=verbose) for x in tqdm(file_names, 'Processing files.')]
    files_data = spawn(read_and_zero_pad, file_names, n_jobs=n_jobs, max_slen=max_slen, sr=sr, verbose=verbose)
    data_dict  = {fnames_we[indx]:files_data[indx][0] for indx,_ in enumerate(tqdm(file_names, 'Processing files.'))}
    return data_dict
# enddef

# Calculate maximum duration of any file in the dataset.
def get_max_slen(paths, round=True):
    file_names  = get_txt_files(get_filenames_without_extension(paths, full_path=True))
    mslen_list  = []
    for file_name in tqdm(file_names, 'Processing files.'):
        df_t = pd.read_csv(file_name, sep='\t', names=['start', 'end', 'crackles', 'wheezes'])
        mslen_list.append(df_t['end'].max())
    # endfor

    m_slen = np.array(mslen_list).max()
    return math.ceil(m_slen) if round else m_slen
# enddef

# Calculate maximum number of seconds for all the files
def get_slen_for_all_files(paths, round=True):
    file_names  = get_txt_files(get_filenames_without_extension(paths, full_path=True))
    mslen_list  = []
    for file_name in tqdm(file_names, 'Processing files.'):
        df_t = pd.read_csv(file_name, sep='\t', names=['start', 'end', 'crackles', 'wheezes'])
        mslen_list.append(df_t['end'].max())
    # endfor

    df = pd.DataFrame(index=[os.path.splitext(os.path.basename(x))[0] for x in file_names])
    df['slen'] = mslen_list
    return df
# enddef

# Remove rare classes
# Asthama and LRTI have very few samples, thus we remove them and don't include them in the
# training process
def remove_rare_classes(data_set):
    features, labels = data_set
    # delete the very rare diseases
    features1 = np.delete(features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0) 
    labels1   = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
    return (features1, labels1)
# enddef

###########################################################
# Populate dataset (training/validation if specified) before training.
# @args
#    paths          => paths dictionary from get_dataset_dirs()
#    sr             => sample rate
#    remove_rare    => Remove rare classes (Asthma & LRTI)
#    num_files      => Only for debugging and verifying that the data pipeline is
#                      working OK. Load first "num_files" files.
#    n_jobs         => Number of jobs to run in parallel
#    split_dataset  => Split dataset to training/validation
#    raw_audio_file => Dump raw audio to this file (and also load if it exists)
#    feat_fn        => Feature extractor function.
#    labels_to_onehot   => Convert labels to 1-hot
#    test_size_ratio    => train/val split ratio
#    **kwargs           => Extra arguments to feature extractor function
# @ returns
#    when split_dataset=True,     (x_train, y_train, x_test, y_test)
#    when split_dataset=False,    (x, y, z) where x is features, y is labels and z is filenames
def populate_dataset(paths, sr=22050, remove_rare=True, num_files=None, n_jobs=4, split_dataset=True,
        raw_audio_file=None, labels_to_onehot=True, test_size_ratio=0.2, feat_fn=e_mel_spectrogram, **kwargs):
    file_names    = get_filenames_without_extension(paths)
    num_files     = len(file_names) if (num_files is None) or (num_files > len(file_names)) else num_files
    file_names    = file_names[:num_files]

    p_id_in_file  = get_patient_ids(file_names)
    p_diag        = pd.read_csv(paths['patient_diag_file'], names=['pId', 'diagnosis'])
    labels        = np.array([p_diag[p_diag['pId'] == x]['diagnosis'].values[0] for x in p_id_in_file])

    # If this file exists, load raw data from here (thus saving a lot of time)
    if raw_audio_file and os.path.isfile(raw_audio_file):
        print('Loading raw audio data from {}'.format(raw_audio_file))
        files_data_d = joblib.load(raw_audio_file)
        files_data   = [files_data_d[x] for x in file_names]
    else:
        files_data_d  = read_all_files(paths, max_slen=30, sr=sr, file_names=file_names, n_jobs=n_jobs)
        files_data    = [files_data_d[x] for x in file_names]
        # Save to file
        if raw_audio_file:
            print('Dumping raw audio data to {}'.format(raw_audio_file))
            joblib.dump(files_data_d, raw_audio_file, protocol=2)
        # endif
    # endif

    features      = np.array(spawn(feat_fn, files_data, n_jobs=n_jobs, sr=sr, **kwargs))
    features      = np.reshape(features, (*features.shape, 1))

    if remove_rare:
        features, labels = remove_rare_classes((features, labels))
    # endif

    # convert labels to 1-hot
    le            = LabelEncoder()
    i_labels      = le.fit_transform(labels)
    oh_labels     = to_categorical(i_labels)

    # Target labels
    tgt_labels = oh_labels if labels_to_onehot else labels

    # train test split
    if split_dataset:
        x_train, x_test, y_train, y_test = train_test_split(features, tgt_labels, stratify=oh_labels, test_size=test_size_ratio, random_state = 42)
        return x_train, y_train, x_test, y_test
    else:
        return (features, tgt_labels, file_names)
    # endif
# enddef

# Convert spectrogram dataset to images and save in save_dir
def convert_dataset_to_images(dataset, save_dir, sr=22050, hop_length=512):
    features, labels, filenames = dataset
    mkdir(save_dir)

    # Create all label sibdirs
    subdirs = list(np.unique(labels))
    for k in subdirs:
        mkdir('{}/{}'.format(save_dir, k))
    # endfor

    for indx, lbl_t in enumerate(tqdm(labels, 'Saving spectrograms.')):
        librosa.display.specshow(np.squeeze(features[indx]), sr=sr, hop_length=hop_length)
        plt.savefig('{}/{}/{}.png'.format(save_dir, lbl_t, filenames[indx]))
        plt.close()
    # endfor
# enddef

# Generate train/test split for the dataset.
def generate_train_test_split(data_set):
    le = LabelEncoder()
    i_labels = le.fit_transform(data_set[1])
    oh_labels = to_categorical(i_labels) 

    # add channel dimension for CNN
    features1 = data_set[0]
    features1 = np.reshape(features1, (*features1.shape,1)) 

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(features1, oh_labels, stratify=oh_labels, test_size=0.2, random_state = 42)

    return x_train, y_train, x_test, y_test
# enddef

###################################################################
# Simple CNN mode for training on MFCC features
def create_cnn_model(num_classes, input_size=(40, 862), num_channels=1):
    filter_size = 2
    
    # Construct model 
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=filter_size,
                     input_shape=(*input_size, num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=32, kernel_size=filter_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=64, kernel_size=filter_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=128, kernel_size=filter_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(num_classes, activation='softmax')) 
    return model
# enddef

def compile_model(model):
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model
# enddef

################################################
# Train simple CNN on MFCC coefficients
def train_model_mfcc(data_set, num_epochs=250, ckpt_dir='checkpoints'):
    # Get training/test dataset
    x_train, y_train, x_test, y_test = data_set

    # model
    model = create_cnn_model(y_train.shape[1], input_size=x_train.shape[1:3])
    model = compile_model(model)
    ckptd = ckpt_dir
    mkdir(ckptd)

    # Start training
    callbacks = [
        ModelCheckpoint(
            filepath='{}/mymodel2_{{epoch:02d}}.h5'.format(ckptd),
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1)
    ]
    start = datetime.now()

    model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=callbacks, verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    return model
# enddef

######################################
# Evaluate results for simple CNN
def eval_and_generate_results(model, data_set):
    x_train, y_train, x_test, y_test = data_set

    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: ", score[1])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: ", score[1])

    preds       = model.predict(x_test) # label scores 
    classpreds  = np.argmax(preds, axis=1) # predicted classes 
    y_testclass = np.argmax(y_test, axis=1) # true classes
    n_classes   = y_test.shape[1] # number of classes

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI']
    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve for Each Class')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], linewidth=3, label='ROC curve (area = %0.2f) for %s' % (roc_auc[i], c_names[i]))
    ax.legend(loc="best", fontsize='x-large')
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()

    # Classification Report
    print(classification_report(y_testclass, classpreds, target_names=c_names))
    # Confusion Matrix
    print(confusion_matrix(y_testclass, classpreds))
# enddef

#######################################################
# populate dataset for image based CNN
# @args
#    paths              => paths dictionary from get_dataset_dirs()
#    spectrogram_path   => Path where spectogram images are saved.
#    sr                 => sample rate
#    remove_rare        => remove rare classes
#    n_jobs             => number of parallel jobs to run
#    test_size_ratio    => train/val split ratio
# @returns
#    (train_gen, val_gen) => Keras ImageDataGenerators for train & val
#
# NOTE: A temporary directory called spectrogram_path + '_final' is created
#       to store dataset after applying train/val split.
def populate_dataset_spectrogram_images(paths, spectrogram_path='./spectrogram_images',
        sr=22050, remove_rare=True, n_jobs=4, test_size_ratio=0.2):
    assert test_size_ratio <= 1.0 and test_size_ratio >= 0.0, 'test_size_ration should be between 0 & 1.'

    target_size = (224, 224)
    batch_size  = 32
    final_spectrogram_path = '{}_final'.format(spectrogram_path)
    train_path             = '{}/train'.format(final_spectrogram_path)
    val_path               = '{}/val'.format(final_spectrogram_path)

    # Delete all dirs
    shutil.rmtree(spectrogram_path, ignore_errors=True)
    shutil.rmtree(final_spectrogram_path, ignore_errors=True)

    # Generate spectrograms
    spectrogram_dataset = populate_dataset(paths, sr=sr, remove_rare=True, num_files=None,
            n_jobs=n_jobs, split_dataset=False, raw_audio_file=None, labels_to_onehot=False,
            feat_fn=e_mel_spectrogram2)
    # Save spectrogram to images
    convert_dataset_to_images(spectrogram_dataset, spectrogram_path, sr=sr)

    # Split dataset
    split_folders.ratio(spectrogram_path, output=final_spectrogram_path, ratio=(1-test_size_ratio, test_size_ratio))
   
    train_gen_p = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
    val_gen_p   = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
    train_gen   = train_gen_p.flow_from_directory(train_path, target_size=target_size, batch_size=batch_size)
    val_gen     = val_gen_p.flow_from_directory(val_path, target_size=target_size, batch_size=batch_size)

    return train_gen, val_gen
# enddef

#############################################################
# Train a usual Image based CNN (we can use VGG16, VGG19 etc) on power melspectrograms
# treating them as images
def train_model_mspectrogram(dataset, num_epochs=50, model_name='vgg19', ckpt_dir='checkpoints2', learning_rate=0.00001):
    # Get training/test dataset
    train_gen, val_gen = dataset

    sample_lbl  = val_gen.next()[1]
    num_classes = sample_lbl.shape[1]

    # model
    model = tf2_models.get_model_fn_map()[model_name](num_classes, weights='imagenet')
    opt   = optimizers.Adam(learning_rate)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    ckptd = ckpt_dir
    shutil.rmtree(ckptd, ignore_errors=True)
    mkdir(ckptd)

    # Start training
    callbacks = [
        ModelCheckpoint(
            filepath='{}/mymodel2_{{epoch:02d}}.h5'.format(ckptd),
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1)
    ]
    start = datetime.now()

    model.fit(train_gen, epochs=num_epochs, validation_data=val_gen, callbacks=callbacks, verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    return model
# enddef
