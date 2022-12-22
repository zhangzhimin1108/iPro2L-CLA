import keras.optimizers
from keras import layers,optimizers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,average_precision_score
from sklearn import metrics
from keras.layers import *
import keras.backend as K
K.set_image_data_format('channels_last')
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
from keras.engine.topology import Layer
from keras.layers import Input, Dropout, Flatten, Dense, BatchNormalization,LSTM
from keras.models import Model
from capsulelayers import CapsuleLayer, PrimaryCap
from layer import Self_Attention
from keras.callbacks import ModelCheckpoint
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
acc_score = []
auc_score = []
sn_score = []
sp_score = []
mcc_score = []
aupr_score = []

def open_fa(file):
    record = []
    f = open(file, 'r')
    for item in f:
        if '>' not in item:
            record.append(item[0:-1])
    f.close()
    return record

def onehot(sequence):
    data = []
    for seq in sequence:
        num= []
        for pp in seq:
            if pp == 'A':
                num.append([1, 0, 0, 0])
            if pp == 'C':
                num.append([0, 1, 0, 0])
            if pp == 'G':
                num.append([0, 0, 1, 0])
            if pp == 'T':
                num.append([0, 0, 0, 1])
        data.append(num)
    return data

seq1 = open_fa('D:/PycharmProjects/pythonProject/iPromoter-CLA/data/data.txt')
seq1_onehot=onehot(seq1)
X=np.array(seq1_onehot)
Y= np.loadtxt('D:/PycharmProjects/pythonProject/iPromoter-CLA/data/labels.txt')
SINGLE_ATTENTION_VECTOR = False
INPUT_DIM = 4
TIME_STEPS = 10
name='promoter_81_58'

for i,(train, test) in enumerate(kfold.split(X, Y)):
    print('\n\n%d'%i)
    #print(i, (train, test))
    path = 'D:/PycharmProjects/pythonProject/iPromoter-CLA/model.h5/%sModel%d.h5' % (name, i)
    checkpoint = ModelCheckpoint(filepath=path,monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='auto')
    def get_model():
        K.clear_session()
        inputs = Input(shape=(81, 4))
        x = Conv1D(64, 5, activation='relu', padding='valid',kernel_initializer='he_normal')(inputs)
        x = Dropout(0.5)(x)
        x = Conv1D(64, 3,activation='relu', padding='valid',kernel_initializer='he_normal')(x)
        x = Dropout(0.5)(x)
        x = Conv1D(32, 3,activation='relu', padding='valid',kernel_initializer='he_normal')(x)
        x = Dropout(0.5)(x)
        x = PrimaryCap(x, dim_capsule=8, n_channels=6, kernel_size=20, strides=1, padding='valid',kernel_initializer='he_normal')
        x = Dropout(0.3)(x)
        x = CapsuleLayer(num_capsule=10, dim_capsule=16,routings=3,kernel_initializer='he_normal')(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        x = Self_Attention(32)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu',kernel_initializer='he_normal')(x)
        x = Dense(32, activation='relu',kernel_initializer='he_normal')(x)
        output = Dense(1, activation='sigmoid',kernel_initializer='glorot_normal')(x)
        model = Model(inputs, output)
        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0005), metrics=['accuracy'])
        return model

    print('Train...')


    callbacks_list = checkpoint
    back = EarlyStopping(monitor='val_loss', patience=30, verbose=2, mode='auto')
    model = None
    model = get_model()
    history=model.fit(X[train], Y[train], epochs=200, batch_size=32, validation_data=(X[test], Y[test]), shuffle=True,callbacks=[callbacks_list, back], verbose=2)
    model.load_weights(path)
    predict_acc_1 = model.predict(X[test])
    predict_acc_2 = []
    for i in predict_acc_1:
        predict_acc_2.append(i[0])
    predict_lable =[]
    for i in predict_acc_2:
        if i>0.5:
             predict_lable.append(1)
        else:
             predict_lable.append(0)
    predict_lable = np.array( predict_lable)
    tn, fp, fn, tp = confusion_matrix(Y[test], predict_lable).ravel()
    sn = tp/(tp+fn)
    sp = tn/(tn+fp)
    mcc= (tp*tn-fp*fn)/(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)
    sn_score.append(sn)
    sp_score.append(sp)
    mcc_score.append(mcc)
    ###########################
    predict_test_y = model.predict(X[test])
    test_auc = metrics.roc_auc_score(Y[test], predict_test_y)
    test_aupr = average_precision_score(Y[test], predict_test_y)
    auc_score.append(test_auc)
    aupr_score.append(test_aupr)
    print("test_auc: ", test_auc)
    print("test_aupr: ", test_aupr)
    score,acc = model.evaluate(X[test], Y[test])
    acc_score.append(acc)
    print('Test score:', score)
    print('Test accuracy:', acc)

print(' final result ')
print(acc_score,auc_score)
mean_acc = np.mean(acc_score)
mean_auc = np.mean(auc_score)
mean_sn = np.mean(sn_score)
mean_sp = np.mean(sp_score)
mean_mcc = np.mean(mcc_score)
mean_aupr = np.mean(aupr_score)
line = 'acc\tsn\tsp\tmcc\tauc\taupr:\n%.2f\t%.2f\t%.2f\t%.4f\t%.4f\t%.4f'%(100*mean_acc,100*mean_sn,100*mean_sp,mean_mcc,mean_auc,mean_aupr)
print('5-fold result:\n'+line)
