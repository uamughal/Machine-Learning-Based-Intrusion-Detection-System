import os
import numpy as np
from keras import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
import tensorflow as tf
from keras.regularizers import l1_l2
from keras.layers import Dropout
from keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, StratifiedKFold
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fusion", help="choose fusion")
parser.add_argument("--png", help="choose png name")
parser.add_argument("--metrics", help="print title name")
args = parser.parse_args()   # arg is variable


def create_model(neurons, act_f, layers, loss_func, ki, dropout_rate,  l1_lambda, l2_lambda, starting_learning_rate):

    model = Sequential()
    model.add(LSTM(int(neurons), kernel_initializer=ki, activation=act_f,
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda),
                   batch_input_shape=(None, WS, feats), return_sequences=True))
    model.add(Dropout(dropout_rate))

    if layers == 2:
        model.add(LSTM(int(neurons * 2/3), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(RepeatVector(WS))
        model.add(LSTM(int(neurons * 2/3), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))

    elif layers == 4:
        model.add(LSTM(int(neurons * 3/4), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons * 2/4), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(RepeatVector(WS))
        model.add(LSTM(int(neurons * 2/4), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons * 3/4), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))

    elif layers == 6:
        model.add(LSTM(int(neurons * 8 / 9), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons * 7 / 9), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons * 6 / 9), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(RepeatVector(WS))
        model.add(LSTM(int(neurons * 6 / 9), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons * 7 / 9), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons * 8 / 9), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))

    elif layers == 8:
        model.add(LSTM(int(neurons * 14 / 16), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons * 12 / 16), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons * 10 / 16), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons * 8 / 16), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(RepeatVector(WS))
        model.add(LSTM(int(neurons * 8 / 16), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons * 10 / 16), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons * 12 / 16), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons * 14 / 16), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(neurons), kernel_initializer=ki, activation=act_f,
                       kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), return_sequences=True))

    model.add(TimeDistributed(Dense(feats)))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=starting_learning_rate,
                                                                 decay_steps=10000,
                                                                 decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(loss=loss_func, optimizer=optimizer, metrics=['MeanSquaredError', 'mean_absolute_error',
                                                                'AUC', 'accuracy', 'Precision', 'Recall'])

    return model


scores = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]


# Define the list of columns to be dropped from the dataset
drop_list_c1 = ['frame.time_epoch', 'wlan_radio.signal_strength (dbm)', 'wlan_radio.Noise level (dbm)',
                'wlan_radio.SNR (db)', 'wlan_radio.preamble', 'wlan.frag', 'wlan.qos', 'wlan.qos.priority',
                'wlan.qos.ack', 'wlan.fcs.status']
drop_list_c2 = ['wlan.bssid', 'wlan.ta', 'wlan.wep.key', 'radiotap.antenna_signal', 'radiotap.channel.flags.cck',
                'wlan_radio.channel', 'wlan_radio.frequency', 'frame.number', 'radiotap.channel.flags.ofdm',
                'wlan.fc.type']  # 20/36 removed
drop_list_p = ['timestamp', 'mid', 'x', 'y', 'z', 'vgx', 'vgy', 'vgz', 'templ',
               'temph', 'baro', 'bat', 'pitch', 'roll', 'h', ]   # 15/21 removed
# 20 keep (combined)
keep_list = ['frame.len', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.duration', 'wlan.seq', 'wlan.fc.subtype',
             'wlan.flags', 'wlan.fcs', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length', 'radiotap.signal_quality',
             'wlan_radio.datarate', 'flight_time', 'agx', 'agy', 'agz', 'yaw', 'tof']
keep_list_c = ['frame.len', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.duration', 'wlan.seq', 'wlan.fc.subtype',
               'wlan.flags', 'wlan.fcs', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length', 'radiotap.signal_quality',
               'wlan_radio.datarate']
keep_list_p = ['flight_time', 'agx', 'agy', 'agz', 'yaw', 'tof', 'pitch', 'vgz']


# path = os.getcwd()
# fusion = "combined"   # fusion: combined, drone1, drone2, cyber, physical
fusion = args.fusion   # fusion: combined, drone1, drone2, cyber, physical
WS = 10   # window size

df = pd.read_csv("../../dataset_central/standardized_dataset_central_{}.csv".format(fusion))
df = df.set_index(['target'])
target = df['target.value']
df = df.drop(columns=['target.value'], axis=1)
# df = df.drop(drop_list_p, axis=1)

if fusion == "combined":
    df = df[keep_list]
elif fusion == "drone1":
    df = df[keep_list]
elif fusion == "drone2":
    df = df[keep_list]
elif fusion == "cyber":
    df = df[keep_list_c]
elif fusion == "physical":
    df = df[keep_list_p]


feats = df.shape[1]
xs = np.empty((1, WS, df.shape[1]))
ys = np.empty(1)

"""
for i in range(0, (df.values.shape[0] - WS)):
    if i == 0:
        xs = np.reshape(df.values[i:(i + WS)], (1, WS, feats))
        ys = np.reshape(df.values[i:(i + WS)], (1, WS, feats))
    else:
        temp_arr = np.reshape(df.values[i:(i + WS)], (1, WS, feats))
        xs = np.vstack((xs, temp_arr))
        ys = np.vstack((ys, temp_arr))
        # ys = np.hstack((ys, target.values[i+WS])) # .hstack is used in LSTM

X = xs
Y = ys
# X, XX, Y, YY = [np.array(x) for x in train_test_split(xs, ys, shuffle=True)]
# X = np.vstack((X, XX))
# Y = np.hstack((Y, YY))

np.save("np_arrays/{}_X.npy".format(fusion), X)
np.save("np_arrays/{}_Y.npy".format(fusion), Y)
"""
X = np.load("np_arrays/{}_X.npy".format(fusion))
Y = np.load("np_arrays/{}_Y.npy".format(fusion))

kf = KFold(n_splits=5, shuffle=True)
fin_f1 = []
fin_val_f1 = []
fin_loss = []
fin_val_loss = []
i = 0
itera = ''

sacc = 0
spre = 0
srec = 0
sf1 = 0
sauc = 0

# hyper-parameters
ki = 'he_uniform'
neurons = 512
act_f = 'relu'
hiddenlayers = 6
dropout_rate = 0.5
l1_lambda = 0.0001
l2_lambda = 0.0001
starting_learning_rate = 0.0001     #  1e-2
batch_size = 256
epochs = 100


for train, test in kf.split(X, Y):
    if i != 0:
        itera = '_' + str(i)

    model = create_model(neurons, act_f, hiddenlayers, 'mse', ki, dropout_rate, l1_lambda, l2_lambda, starting_learning_rate)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=10, verbose=1, mode='auto',
                                              baseline=None, restore_best_weights=True)

    history = model.fit(x=X[train], y=Y[train], validation_data=(X[test], Y[test]), epochs=epochs, batch_size=batch_size, callbacks=[early_stop]).history

    recall = np.array(history['recall'])
    precision = np.array(history['precision'])
    f1 = (2 * ((precision * recall)/(precision + recall)))

    v_recall = np.array(history['val_recall'])
    v_precision = np.array(history['val_precision'])
    v_f1 = (2 * ((v_precision * v_recall)/(v_precision + v_recall)))

    fin_loss.append(history['loss'])
    fin_val_loss.append(history['val_loss'])

    fin_f1.append(f1.tolist())
    fin_val_f1.append(v_f1.tolist())

    sacc += history['val_accuracy'][len(history['val_accuracy']) - 1]
    spre += history['val_precision'][len(history['val_precision']) - 1]
    srec += history['val_recall'][len(history['val_recall']) - 1]
    sf1 += v_f1[v_f1.shape[0] - 1]
    sauc += history['val_auc'][len(history['val_auc']) - 1]

    i += 1


# save the model
# model.save(f"trained_models/model_{fusion}.h5")

length = max(map(len, fin_f1))
y = np.array([xi+[np.NaN]*(length-len(xi)) for xi in fin_f1])

length = max(map(len, fin_val_f1))
y2 = np.array([xi+[np.NaN]*(length-len(xi)) for xi in fin_val_f1])

length = max(map(len, fin_loss))
y3 = np.array([xi+[np.NaN]*(length-len(xi)) for xi in fin_loss])

length = max(map(len, fin_val_loss))
y4 = np.array([xi+[np.NaN]*(length-len(xi)) for xi in fin_val_loss])


train_scores_mean = np.nanmean(y, axis=0)
train_scores_std = np.nanstd(y, axis=0)
test_scores_mean = np.nanmean(y2, axis=0)
test_scores_std = np.nanstd(y2, axis=0)


x_ticks = np.arange(length)
title = r"Learning Curves (FNN, 5 layers, 256 neurons, relu, glorot)"
_, axes = plt.subplots(1, 1, figsize=(14, 6))
axes.grid()
axes.fill_between(
    x_ticks,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.1,
    color="r",
)
axes.fill_between(
    x_ticks,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.1,
    color="g",
)

# axes.plot(x_ticks, train_scores_mean, 'o-', color='r', label='Train', linewidth=3)  # linewidth
# axes.plot(x_ticks, test_scores_mean, 'o-', color='g', label='Validation', linewidth=3)
axes.plot(x_ticks, train_scores_mean, 'o-', color='r', label='Train')
axes.plot(x_ticks, test_scores_mean, 'o-', color='g', label='Validation')
axes.set_ylabel('F1')
axes.set_xlabel('Epoch')
axes.legend(loc='upper left')
# plt.savefig("AE_results/AE_F1_{}.png".format(fusion))
plt.savefig("AE_results/AE_F1_reg_{}".format(args.png))
# plt.show()


train_scores_mean = np.nanmean(y3, axis=0)
train_scores_std = np.nanstd(y3, axis=0)
test_scores_mean = np.nanmean(y4, axis=0)
test_scores_std = np.nanstd(y4, axis=0)

x_ticks = np.arange(length)
# title = r"Learning Curves (FNN, 5 layers, 256 neurons, relu, glorot)"
_, axes = plt.subplots(1, 1, figsize=(14, 6))
axes.grid()
axes.fill_between(
    x_ticks,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.1,
    color="r",
)
axes.fill_between(
    x_ticks,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.1,
    color="g",
)

axes.plot(x_ticks, train_scores_mean, 'o-', color="r", label='Train')
axes.plot(x_ticks, test_scores_mean, 'o-', color="g", label='Validation')
axes.set_ylabel('Loss')
axes.set_xlabel('Epoch')
axes.legend(loc='upper left')
# plt.savefig("AE_results/AE_loss_reg_{}.png".format(fusion))
plt.savefig("AE_results/AE_loss_reg_{}".format(args.png))
# plt.show()


sacc = sacc / 5
spre = spre / 5
srec = srec / 5
sf1 = sf1 / 5
sauc = sauc / 5

list_results = [sacc, spre, srec, sf1, sauc]

print(f"central_{args.metrics}")  # --metrics= combined
# print(f"central_{fusion}")
print('accuracy, precision, recall, f1, auc :', list_results)



"""
# Save the results as a CSV file
results_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "AUC",
        ],
        "Value": list_results,
    }
)
results_df.to_csv(f"results/results_{fusion}.csv", index=False)
"""


"""
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.grid()
ax.plot(history['loss'], 'r', label='Train', linewidth=3)
ax.plot(history['val_loss'], 'g', label='Validation', linewidth=3)
#ax.set_title('LSTM-AE -- {} data -- window size of {}'.format(fusion, WS), fontsize=16)
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
#plt.savefig(path + "/Figures/FINAL/AE_{}.png".format(fusion))
plt.show()
"""
