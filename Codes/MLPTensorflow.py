import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import time

tf.disable_v2_behavior()


def filterData():
    df = pd.read_csv('data.csv')
    print("how large the data sould be?")
    data_size = input()


    data = df.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'number_outpatient',
                    'number_inpatient'], axis=1)
    data = data.replace(
        ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
        [5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
    data = data.replace(["Up", "Down", "Ch", "Steady", "Yes", "No"], [3, 0, 1, 2, 1, 0])
    data = data.replace(["None", "Normal", "Norm", ">200", ">300"], [0, 1, 1, 2, 3])
    data = data.replace([">7", ">8"], [2, 3])
    data = data.replace(["NO", "<30", ">30"], [0, 1, 2])
    data = pd.get_dummies(data, columns=['race', 'gender', 'admission_source_id', 'discharge_disposition_id', 'admission_source_id', 'diag_1', 'diag_2', 'diag_3'])

    if data_size=="all":
        data_size=len(data)
    data_size = int(data_size)


    data = data[:data_size]

    data_train = data[:round(len(data)*7/10)]
    data_verif = data[round(len(data)*7/10):]

    print("training set length : "+str(len(data_train)))
    print("verification set length : "+str(len(data_verif)))

    label_train1 = data_train[['readmitted']]
    label_verif1 = data_verif[['readmitted']]

    data_train = data_train.drop(['readmitted'], axis=1)
    data_verif = data_verif.drop(['readmitted'], axis=1)

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

    return data_train , label_train ,data_verif ,label_verif

def train_model():
    data_x = data_train
    data_y = label_train
    print("start training the model")
    start_time = time.time()
    for i in range(0, 1000):
        sess.run(update, feed_dict={x: data_x, y_: data_y})
    print("finish training the model")
    print("--- %s seconds ---" % round(time.time() - start_time))



def verification():
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: data_verif, y_: label_verif})
    print("training length : ", len(data_train), "/ data check : ", len(data_verif))
    print("Loss: {:.3f}\tAccuracy: {:.2%}".format(loss, acc))


data_train,label_train ,data_verif , label_verif= filterData()

features = len(data_train[0])
print (features)
(hidden1_size, hidden2_size,hidden3_size,hidden4_size) = (1000, 300, 200, 75)

categories = 3

x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, categories])

W1 = tf.Variable(tf.truncated_normal([features, hidden1_size], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[hidden1_size]))
z1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[hidden2_size]))
z2 = tf.nn.relu(tf.matmul(z1, W2) + b2)

W3 = tf.Variable(tf.truncated_normal([hidden2_size, hidden3_size], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[hidden3_size]))
z3 = tf.nn.relu(tf.matmul(z2, W3) + b3)

W4 = tf.Variable(tf.truncated_normal([hidden3_size, hidden4_size], stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, shape=[hidden4_size]))
z4 = tf.nn.relu(tf.matmul(z3, W4) + b4)

W5 = tf.Variable(tf.truncated_normal([hidden4_size, categories], stddev=0.1))
b5 = tf.Variable(tf.constant(0.1, shape=[categories]))

y = tf.nn.softmax(tf.matmul(z4, W5) + b5)

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
cross_entropy = tf.reduce_mean(loss)
update = tf.train.AdamOptimizer().minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


train_model()
verification()
