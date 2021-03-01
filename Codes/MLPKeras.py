import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import time
from keras.callbacks import EarlyStopping

from sklearn.ensemble import RandomForestClassifier
from yellowbrick.model_selection import FeatureImportances

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import classification_report


def filterData():
    df = pd.read_csv(r'diabetic_data.csv')
    print("how large the data sould be?")
    print("0<100000 or \"all\" for all the data")
    data_size = input()

    # 'metformin' , 'repaglinide' , 'nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone'
    # ,'acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone'
    # ,'metformin-rosiglitazone','metformin-pioglitazone'

    # 'citoglipton' , 'metformin-pioglitazone' , 'metformin-rosiglitazone' , 'glimepiride-pioglitazone' ,
    # 'examide' , 'acetohexamide' , 'troglitazone' , 'glipizide-metformin' , 'tolbutamide'

    data = df.drop(['encounter_id', 'patient_nbr', 'payer_code' ], axis=1)

    # dataSelected = data[['diag_1' , 'diag_2' , 'diag_3' ,'discharge_disposition_id', 'medical_specialty', 'number_outpatient',
    #                 'number_inpatient',  'admission_source_id', 'readmitted']]
    # dataSelected = pd.get_dummies(dataSelected , columns=['diag_1' , 'diag_2' , 'diag_3','discharge_disposition_id', 'medical_specialty', 'number_outpatient',
    #                 'number_inpatient',  'admission_source_id'])

    data = data.drop(['diag_1', 'diag_2', 'diag_3'] , axis=1)

    data = data.replace(
        ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
        [5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
    data = data.replace(
        ["[0-25)", "[25-50)", "[50-75)", "[75-100)", "[100-125)", "[125-150)","[150-175)","[175-200)",">200"],
        [13,38,63,88,113,138,163,188,205])
    data = data.replace(["Up", "Down", "Ch", "Steady", "Yes", "No","?"], [2, -1, 1, 1, 1, 0, 0])
    data = data.replace(["None", "Normal", "Norm", ">200", ">300"], [0, 1, 1, 2, 3])
    data = data.replace([">7", ">8"], [1, 2])
    data = data.replace(["NO", "<30", ">30"], [0, 1, 2])
    # dataSelected = dataSelected.replace(["NO", "<30", ">30"], [0, 1, 2])
    # data = pd.get_dummies(data, columns=['race','gender','discharge_disposition_id', 'medical_specialty', 'number_outpatient',
    #                 'number_inpatient',  'admission_source_id','diag_1', 'diag_2', 'diag_3',
    #                                      'metformin-rosiglitazone','metformin-pioglitazone','acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone'
    #                                      ,'metformin' , 'repaglinide' , 'nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone'])



    data = pd.get_dummies(data, columns=['race','gender','discharge_disposition_id', 'medical_specialty', 'number_outpatient',
                    'number_inpatient',  'admission_source_id'])



    if data_size=="all":
        data_size=len(data)
    data_size = int(data_size)

    strFeat=''
    for i in range(len(data.columns)):
        strFeat = strFeat+"\n"+str(i)+'-'+str(data.columns[i])

    data = data[:data_size]
    # dataSelected = dataSelected[:data_size]

    data_train = data[:round(len(data)*7/10)]
    data_verif = data[round(len(data)*7/10):]

    # dataSelected_train = dataSelected[:round(len(data)*7/10)]
    # dataSelected_verif = dataSelected[round(len(data)*7/10):]

    class0len = len(data_train.loc[(data_train['readmitted'] == 0)])
    dataClass1 = len(data_train.loc[(data_train['readmitted'] == 1)])


    print("1-Normal\n2-oversampling\n3-undersampling")
    over= input()
    if over=='2':
        print("before oversampling")
        print("class 0 cases : " , len(data_train.loc[(data_train['readmitted'] == 0)]))
        print("class 1 cases : " , len(data_train.loc[(data_train['readmitted'] == 1)]))
        print("class 2 cases : " , len(data_train.loc[(data_train['readmitted'] == 2)]))
        dataClass1 = data_train.loc[(data_train['readmitted'] == 1)]
        while len(data_train.loc[(data_train['readmitted'] == 1)]) < class0len:
            data_train = pd.concat([data_train , dataClass1] , ignore_index=True , sort=False)
        print("\nafter oversampling")
        print("class 0 cases : " , len(data_train.loc[(data_train['readmitted'] == 0)]))
        print("class 1 cases : " , len(data_train.loc[(data_train['readmitted'] == 1)]))
        print("class 2 cases : " , len(data_train.loc[(data_train['readmitted'] == 2)]))

    if over=='3':
        print("before undersampling")
        print("class 0 cases : " , len(data_train.loc[(data_train['readmitted'] == 0)]))
        print("class 1 cases : " , len(data_train.loc[(data_train['readmitted'] == 1)]))
        print("class 2 cases : " , len(data_train.loc[(data_train['readmitted'] == 2)]))
        train0 = data_train.loc[(data_train['readmitted'] == 0)].head(dataClass1)
        train2 = data_train.loc[(data_train['readmitted'] == 2)].head(dataClass1)
        train1 = data_train.loc[(data_train['readmitted'] == 1)].head(dataClass1)


        data_train = pd.concat([train0 , train2] , ignore_index=True , sort=False)
        data_train = pd.concat([data_train , train1] , ignore_index=True , sort=False)

        print("\nafter undersampling")
        print("class 0 cases : " , len(data_train.loc[(data_train['readmitted'] == 0)]))
        print("class 1 cases : " , len(data_train.loc[(data_train['readmitted'] == 1)]))
        print("class 2 cases : " , len(data_train.loc[(data_train['readmitted'] == 2)]))

    else:
        pass






    print("\ntraining set length : " + str(len(data_train)))
    print("verification set length : " + str(len(data_verif)))

    label_train1 = data_train[['readmitted']]
    label_verif1 = data_verif[['readmitted']]

    data_train = data_train.drop(['readmitted'], axis=1)
    data_verif = data_verif.drop(['readmitted'], axis=1)


###########select
    # dataSelected_train = dataSelected_train.drop(['readmitted'] , axis=1)
    # dataSelected_verif = dataSelected_verif.drop(['readmitted'] , axis=1)
    # selector = SelectKBest(f_classif , k=50)
    # dataSelected_train = selector.fit_transform(dataSelected_train.to_numpy() , label_train1.to_numpy())
    # dataSelected_verif = selector.fit_transform(dataSelected_verif.to_numpy() , label_verif1.to_numpy())
    #
    # data_train = np.append(data_train.to_numpy() , dataSelected_train , axis=1)
    # data_verif = np.append(data_verif.to_numpy() , dataSelected_verif , axis=1)



    data_train = data_train.to_numpy()
    data_verif = data_verif.to_numpy()

    label_train1 = label_train1.to_numpy()
    label_verif1 = label_verif1.to_numpy()

    label_train = np.zeros((len(label_train1), 3))
    label_verif = np.zeros((len(label_verif1), 3))

    for i in range(len(label_train1)):
        if label_train1[i][0] == 0:
            label_train[i] = np.array([1, 0, 0])
        elif label_train1[i][0] == 1:
            label_train[i] = np.array([0, 1, 0])
        elif label_train1[i][0] == 2:
            label_train[i] = np.array([0, 0, 1])

    for i in range(len(label_verif1)):
        if label_verif1[i][0] == 0:
            label_verif[i] = np.array([1, 0, 0])
        elif label_verif1[i][0] == 1:
            label_verif[i] = np.array([0, 1, 0])
        elif label_verif1[i][0] == 2:
            label_verif[i] = np.array([0, 0, 1])


    return data_train , label_train ,data_verif ,label_verif,strFeat

def getPlotGraphs():
    y_pred = model.predict(X_val , verbose=1)
    y_pred_bool = np.argmax(y_pred , axis=1)
    y_val_bool = np.argmax(y_val , axis=1)
    print(classification_report(y_val_bool , y_pred_bool))

    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1 , epoch + 1)
    plt.plot(epochs , loss_train , 'g' , label='Training loss')
    plt.plot(epochs , loss_val , 'b' , label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    loss_train = history.history['accuracy']
    loss_val = history.history['val_accuracy']
    epochs = range(1 , epoch + 1)
    plt.plot(epochs , loss_train , 'g' , label='Training accuracy')
    plt.plot(epochs , loss_val , 'b' , label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()




X_train,y_train ,X_val , y_val, strf= filterData()
epoch = 50





print("\n1-train model\n2-evaluate the last trained model\n3-Simple softmax\n4-evaluate the features")
choice = input()

if(choice == '1'):

    features = len(X_train[0])
    print(features)
    categories = 3
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy' ,mode='max', verbose=1)

    model = Sequential()
    # model.add(Dense(round(features*2) , input_dim=features , activation='relu',kernel_regularizer=keras.regularizers.l1(l=0.01)))

    model.add(Dense(round(features) , input_dim=features , activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(round(features) , activation='relu',))
    model.add(Dense(round(features*3/4) , activation='relu',))
    # model.add(Dropout(0.2))


    model.add(Dense(round(features*1/3) , activation='relu',))
    # model.add(Dropout(0.2))



    model.add(Dense(3, activation='softmax'))

    opt =keras.optimizers.Adam(lr=0.01 , beta_1=0.9 , beta_2=0.999 , epsilon=1e-08 , decay=0.0)
    # es = EarlyStopping(monitor='loss' , mode='min' , verbose=1)
    model.compile(optimizer=opt ,
                  loss='categorical_crossentropy' ,
                  metrics=['accuracy'],
                  )




    # history = model.fit(X_train , y_train , epochs=epoch , batch_size=2000, validation_data=(X_val , y_val),callbacks=[es])
    history = model.fit(X_train , y_train , epochs=epoch , batch_size=2000, validation_data=(X_val , y_val))

    loss , acc = model.evaluate(X_val , y_val , verbose=2)
    print("trained model, accuracy: {:5.2f}%".format(100 * acc))

    getPlotGraphs()
    model.save(r'MLPModel')





if choice== '2' :
    model = keras.models.load_model(r'MLPModel')
    print(model.predict(X_train))

    loss , acc = model.evaluate(X_val , y_val , verbose=2)
    print("trained model, accuracy: {:5.2f}%".format(100 * acc))
    y_pred = model.predict(X_val , verbose=1)
    y_pred_bool = np.argmax(y_pred , axis=1)
    y_val_bool = np.argmax(y_val , axis=1)
    print(classification_report(y_val_bool , y_pred_bool))

if choice =='4':
    features = len(X_train[0])
    print(features)
    print(strf)
    categories = 3

    model = RandomForestClassifier()
    # Visualizing Feature Importance
    viz = FeatureImportances(model)
    start_time = time.time()
    print("start")
    viz.fit(X_train , y_train)
    print("--- %s seconds ---" % round(time.time() - start_time))
    viz.show()

if choice =='3':
    features = len(X_train[0])
    print(features)
    categories = 3

    model = Sequential()
    model.add(Dense(3 , input_dim=features , activation='softmax',kernel_regularizer=keras.regularizers.l1(l=0.1)))  # softmax

    model.compile(optimizer='adam' ,
                  loss='categorical_crossentropy' ,
                  metrics=['accuracy'])
    history = model.fit(X_train , y_train , epochs=epoch , batch_size=1024 , validation_data=(X_val , y_val))

    loss , acc = model.evaluate(X_val , y_val , verbose=2)
    print("trained model, accuracy: {:5.2f}%".format(100 * acc))

    getPlotGraphs()