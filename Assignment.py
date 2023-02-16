import numpy
import pandas as pd
import sklearn
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Flatten
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import keras_tuner
from keras_tuner import HyperModel
from sklearn import preprocessing
import plotly.express as px
from sklearn.model_selection import StratifiedKFold
import numpy as np
from perceptron import Perceptron
from leastSquares import LSquares


def scale_data(scaler, data):
    '''Scales the data given using the given scaler'''

    return scaler.fit_transform(data)

def visualize(*columns):
    '''Visualizes the frequency histogram of column given.
    If more than one column given, visualize their relationship with scatterplot'''

    if len(columns) == 2:
        px.histogram(columns[0], x=columns[1].name).show()
    elif len(columns) == 3:
        px.scatter(columns[0], x=columns[1].name, y=columns[2].name).show()
    elif len(columns) == 4:
        px.scatter(columns[0], x=columns[1].name, y=columns[2].name, color=columns[3].name).show()
    else:
        px.scatter(columns[0], x=columns[1].name, y=columns[2].name, color=columns[3].name, size=columns[4].name).show()

def replace_nan(x):
    '''Replaces NaN with the median value of a column'''

    col_num = x.shape[1]
    for i in range(col_num):
        if np.isnan(x[:, i]).any():
            x[np.isnan(x[:, i]), i] = numpy.nanmedian(i)
    return x

df = pd.read_csv("housing.csv")

while True:
    inp = input("Type the name of a feature to view its frequency histogram, or more (use ',' to separate them) in order to view their relationships. Press 1 to exit\n")
    if inp == '1':
        break
    lst = [i.strip() for i in inp.split(',')]
    flag = True
    for i in lst:
        if i not in df.columns:
             print("ERROR: This feature doesnt exist in this dataset.")
             flag = False
    if flag:
        if len(lst) == 1:
            visualize(df, df[lst[0]])
        elif len(lst) == 2:
            visualize(df, df[lst[0]], df[lst[1]])
        elif len(lst) == 3:
            visualize(df, df[lst[0]], df[lst[1]], df[lst[2]])
        else:
            visualize(df, df[lst[0]], df[lst[1]], df[lst[2]], df[lst[3]])


# finding the numerical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# finding the categorical cols and encoding them
categorical_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
encoder = preprocessing.OneHotEncoder(sparse_output=False)
categ_matrix = encoder.fit_transform(df[categorical_cols])

# choosing one of the below to scale the data. In the end i chose robust scaler because it showed the better performance for this dataset (during training/testing)
mms = preprocessing.MinMaxScaler()
ss = preprocessing.StandardScaler()
rs = preprocessing.RobustScaler()

x_scaled_with_target = scale_data(rs, pd.DataFrame(df, columns=numerical_cols))
y_scaled = x_scaled_with_target[:,-1]
x_scaled_without_categ = np.delete(x_scaled_with_target, -1, axis=1)
x_scaled = np.append(x_scaled_without_categ, categ_matrix, axis=1)

col_of_ones = np.array([[1] for i in range(1, x_scaled.shape[0] + 1)])
x_scaled = np.append(x_scaled,col_of_ones,axis=1)
x_scaled = replace_nan(x_scaled)

# Need the original dataframes for the stratified k-fold. Need this k-fold since we have unbalanced distribution
y = pd.DataFrame(df['median_house_value'], columns=['median_house_value'])
x = df.drop(columns=['median_house_value'], axis=1)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
perc = Perceptron()
ls = LSquares()

for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    print(
        "                                                       CLASSIFYING WITH PERCEPTRON                                                            ",
        '\n')

    x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
    y_train_fold, y_test_fold = y_scaled[train_index], y_scaled[test_index]
    acc , conf_matrix = perc.fit(x_train_fold, y_train_fold)
    print(f"From training sample {i+1}. Accuracy was {acc} and the confusion matrix is:\n{conf_matrix}\n")
    acc, conf_matrix  = perc.test(x_test_fold, y_test_fold)
    print(f"From test sample {i+1}. Accuracy was {acc} and the confusion matrix is:\n{conf_matrix}\n")


# In order to apply the formula for Least Squares, i need to get rid of the OneHot encoding. But i also need the input features, so i encoded each row of ocean_prox regarding where its index of 1 is, then re-scaled
categ_col=[]
for i in categ_matrix:
    for j in range(len(i)):
        if i[j] == 1.:
            categ_col.append([j])
tst = np.delete(np.array(x), [-1], axis=1)
tst = np.append(tst, categ_col, axis=1)
x_scaled = scale_data(rs, tst)
x_scaled = replace_nan(x_scaled)

for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
    y_train_fold, y_test_fold = y_scaled[train_index], y_scaled[test_index]
    tr_mse, tr_mae            =  ls.fit(x_train_fold, y_train_fold)
    print(f"From training sample {i+1}. MSE: {tr_mse}, MAE: {tr_mae} )")
    tst_mse , tst_mae        =  ls.fit(x_test_fold, y_test_fold)
    print(f"From testing sample {i+1}. MSE: {tst_mse}, MAE: {tst_mae}")


# Ι commented out my search for hyperparameters during non-linear regression with neural network (it takes a lot of time)

# class MyHyperModel(HyperModel):
#     def build(self, hp):
#         model2 = Sequential()
#         model2.add(Flatten())
#         for i in range(hp.Int("num_layers", 1, 2)):
#             model2.add(
#                 Dense(
#                     # Tune number of units separately.
#                     units=hp.Int(f"units_{i}", min_value=80, max_value=128, step=4),
#                     activation="relu",
#                 )
#             )
#         if hp.Boolean("dropout"):
#             model2.add(Dropout(rate=0.25))
#
#         model2.add(Dense(1, activation='linear'))
#         model2.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
#
#
#         return model2
#
#
# tunerr = keras_tuner.Hyperband(
#                        hypermodel=MyHyperModel(),
#                        objective ='val_mean_squared_error',
#                        max_epochs=100,
#                         directory="my_dir",
#                         project_name="helloworld",
#                         overwrite=True,
#
# )
#
# stop_early = EarlyStopping(monitor='val_loss', patience=15)
#
# for i, (train_index, test_index) in enumerate(skf.split(x, y)):
#     tunerr.search(x_scaled[train_index], y_scaled[train_index], epochs=10,
#                  validation_data=(x_scaled[test_index], y_scaled[test_index]),
#              batch_size=32, callbacks=[stop_early])
#     best_hps = tunerr.get_best_hyperparameters(num_trials=1)[0]
#     if i == 5:
#         break

# print(tunerr.results_summary(num_trials=2))


model = Sequential()
model.add(Dense(180, activation="relu", input_dim=x_scaled.shape[1]))
model.add(Dense(150, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mean_squared_error'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
lst_train_mse = []
lst_test_mse = []
lst_train_mae = []
lst_test_mae = []

for train_index, test_index in skf.split(x, y):
    model.fit(x_scaled[train_index], y_scaled[train_index], validation_data=(x_scaled[test_index], y_scaled[test_index]), epochs=150, batch_size=100, callbacks=[es])
    PredTrain = model.predict(x_scaled[train_index])
    PredTest = model.predict(x_scaled[test_index])
    lst_train_mse.append(sklearn.metrics.mean_squared_error(PredTrain, y_scaled[train_index]))
    lst_test_mse.append(sklearn.metrics.mean_squared_error(PredTest, y_scaled[test_index]))
    lst_train_mae.append(sklearn.metrics.mean_absolute_error(PredTrain,y_scaled[train_index]))
    lst_test_mae.append(sklearn.metrics.mean_absolute_error(PredTest, y_scaled[test_index]))

for i, (j,k,m,n) in enumerate(zip(lst_train_mse, lst_test_mse, lst_train_mae, lst_test_mae)):
    print(f" From training sample {i+1}. MSE: {j}, MAE: {m}")
    print(f"From testing sample {i+1}. MSE: {k}, MAE: {n}")


