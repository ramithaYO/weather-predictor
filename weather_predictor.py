import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from sklearn import preprocessing
app = Flask(__name__)
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from sklearn.feature_selection import SelectKBest, chi2
CORS(app)
api = Api(app)


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}



def find_correlation(data, threshold=0.9):
    corr_mat = data.corr()
    corr_mat.loc[:, :] = np.tril(corr_mat, k=-1)
    already_in = set()
    result = []
    for col in corr_mat:
        perfect_corr = corr_mat[col][abs(corr_mat[col])> threshold].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat

def predicit(a,b,c,d,e,f,g,h,i,j):
    scaler = preprocessing.MinMaxScaler()
    wa = pd.read_csv("weatherAUS.csv")
    wa2 = wa['Location'] == "Sydney"
    wa3 = wa[wa2]
    data = wa3.drop(columns=['Date', 'Location', 'RISK_MM', 'WindGustSpeed', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'Cloud3pm'])
    data = data.dropna(how='any')
    data['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
    data['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)
    columns_to_drop = find_correlation(data.drop(columns=['RainTomorrow', 'Sunshine']), 0.7)
    df4 = data.drop(columns=columns_to_drop)
    df4 = df4.drop(columns=['WindSpeed3pm'])
    df = df4

    X = df.loc[:, df.columns != 'RainTomorrow']
    y = df[['RainTomorrow']]
    selector = SelectKBest(chi2, k=3)
    selector.fit(X, y)
    X_new = selector.transform(X)
    print(X.columns[selector.get_support(indices=True)])

    t0 = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
    # normalize the train data set
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)

    #add the user input to the test data set
    X_test.loc[62257] = [a, b, c, d, e, f, g, h, i, j]

    # normalize the test data set
    scaler.fit(X_test)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    clf_rf.fit(X_train, y_train)
    y_pred = clf_rf.predict(X_test)
    output = y_pred[661]
    score = 'YES'
    if(output==0):
        score = 'NO'
    # score = accuracy_score(y_test, y_pred)
    # print('Accuracy :', score)
    return score


parser = reqparse.RequestParser()

parser.add_argument('MaxTemp')
parser.add_argument('Rainfall')
parser.add_argument('Evaporation')
parser.add_argument('Sunshine')
parser.add_argument('WindSpeed9am')
parser.add_argument('Humidity9am')
parser.add_argument('Humidity3pm')
parser.add_argument('Pressure3pm')
parser.add_argument('Cloud9am')
parser.add_argument('RainToday')


class getItem(Resource):
    def post(self):
        args = parser.parse_args()
        val=predicit(args['MaxTemp'],args['Rainfall'],args['Evaporation'],args['Sunshine'],args['WindSpeed9am'],args['Humidity9am'],args['Humidity3pm'],args['Pressure3pm'],args['Cloud9am'], args['RainToday'])
        val2=str(val)

        return {'result': val2}, 201




api.add_resource(getItem, '/addItem')
api.add_resource(HelloWorld, '/hello')
if __name__=='__main__':
    app.run()
# image_filename = '1.PNG' # replace with your own image
# encoded_image = base64.b64encode(open(image_filename, 'rb').read())
#
#
# data = pd.read_csv("weatherAUS.csv")
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# data.head(),
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# colors = {
#     'text':'white',
#     'text': '#7FDBFF',
# 'background-image': 'url(https://images.pexels.com/photos/2310641/pexels-photo-2310641.jpeg?auto=format%2Ccompress&cs=tinysrgb&dpr=2&h=650&w=940)'
# }
# app.layout = html.Div(style={'background-image': 'url(https://i0.wp.com/michaelzhao.net/wp-content/uploads/2016/05/hobet_1984-2015.gif?resize=720%2C480)'},children=[
#
#     html.H1(style={'text-align': 'center'},children='weather prediction'),
#
#     html.Div(children='''
#         Dash: A web application framework for Python.
#     '''),
#
#
# ])
#
# if __name__ == '__main__':
#     app.run_server(debug=True)