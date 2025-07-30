from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import numpy as np
import matplotlib.pyplot as mplt
import io
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pickle
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Convolution2D
from math import sqrt
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

global uname

analyzer = SentimentIntensityAnalyzer()

scaler = MinMaxScaler(feature_range = (0, 1))
scaler1 = MinMaxScaler(feature_range = (0, 1))

dataset = pd.read_csv("Dataset/text_ratings.csv")
dataset.fillna(0, inplace=True)#remove missing values
data = dataset.values
Y = data[:,3:4]
X = data[:,2:4]

X = scaler.fit_transform(X)
Y = scaler1.fit_transform(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

dataset = dataset.drop_duplicates(subset=['Video_id'])
dataset = dataset.values

#function to calculate accuracy and prediction sales graph
def calculateMetrics(algorithm, predict, test_labels):
    predict = predict.reshape(-1, 1)
    predict = scaler1.inverse_transform(predict)
    test_label = scaler1.inverse_transform(test_labels)
    predict = predict.ravel()
    test_label = test_label.ravel()
    mse_error = sqrt(mean_squared_error(test_label, predict))
    mse_error = mse_error * 10000 
    return mse_error

#now train & plot CNN crop yield prediction
X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
#training CNN model
cnn_model = Sequential()
cnn_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
cnn_model.add(Flatten())
cnn_model.add(Dense(units = 256, activation = 'relu'))
cnn_model.add(Dense(units = 1))
cnn_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    cnn_model.fit(X_train1, y_train, batch_size = 8, epochs = 1000, validation_data=(X_test1, y_test), callbacks=[model_check_point], verbose=1)
else:
    cnn_model.load_weights("model/cnn_weights.hdf5")
predict = cnn_model.predict(X_test1)
mse_error = calculateMetrics("CNN + LSTM", predict, y_test)#call function to plot LSTM crop yield prediction

def FileComment(request):
    if request.method == 'GET':
       return render(request, 'FileComment.html', {})

def SingleComment(request):
    if request.method == 'GET':
       return render(request, 'SingleComment.html', {})    

def getSentiment(comment):
    sentiment = -1
    vs = analyzer.polarity_scores(comment)
    compound = vs['compound']
    if compound >= 0.5:
        sentiment =  5
    elif compound < 0.5 and compound >= 0.1:
        sentiment = 4
    elif compound < 0.1 and compound >= 0.05:
        sentiment = 3
    elif compound < 0.05 and compound > -0.05:
        sentiment = 2
    else:
        sentiment = 1
    return sentiment, compound

def getRecommendation(sentiment):
    global dataset
    recommendation = []
    for i in range(len(dataset)):
        if dataset[i,3] == sentiment:
            if dataset[i,0] not in recommendation:
                recommendation.append(dataset[i,0])
        if len(recommendation) == 10:
            break
    return recommendation

def SingleCommentAction(request):
    if request.method == 'POST':
        global cnn_model, scaler, scaler1
        cnn_model = load_model("model/cnn_weights.hdf5")
        comment = request.POST.get('t1', False)
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Test Comment</th><th><font size="" color="black">Predicted Sentiment</th>'
        output +='<th><font size="" color="black">Hated %</th><th><font size="" color="black">Recommended Videos</th></tr>'
        sentiment, hatred = getSentiment(comment)#finding hatred percentage
        data = []
        data.append([sentiment, sentiment])
        data = np.asarray(data)
        data = scaler.transform(data)
        data = np.reshape(data, (data.shape[0], data.shape[1], 1, 1))
        predict = cnn_model.predict(data)
        predict = scaler1.inverse_transform(predict)
        predict = predict.ravel()
        predict = predict[0]
        predict = int(round(predict))
        recommend = getRecommendation(predict)
        output+='<td><font size="" color="black">'+comment+'</td><td><font size="" color="black">'+str(predict)+'</td>'
        output +='<td><font size="" color="black">'+str(hatred)+'</td><td><font size="" color="black">'+str(recommend)+'</td></tr>'
        output+= "</table></br>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def FileCommentAction(request):
    if request.method == 'POST':
        global dataset, scaler, scaler1, cnn_model
        cnn_model = load_model("model/cnn_weights.hdf5")
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("SentimentApp/static/"+fname):
            os.remove("SentimentApp/static/"+fname)
        with open("SentimentApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        testData = pd.read_csv("SentimentApp/static/"+fname)
        testData.fillna(0, inplace = True)
        testData = testData.values
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Test Comment</th><th><font size="" color="black">Predicted Sentiment</th><th><font size="" color="black">Recommended Videos</th></tr>'
        result = []
        for i in range(len(testData)):
            comment = testData[i,0]
            sentiment, hatred = getSentiment(comment)
            data = []
            data.append([sentiment, sentiment])
            data = np.asarray(data)
            data = scaler.transform(data)
            data = np.reshape(data, (data.shape[0], data.shape[1], 1, 1))
            predict = cnn_model.predict(data)
            predict = scaler1.inverse_transform(predict)
            predict = predict.ravel()
            predict = predict[0]
            predict = int(round(predict))
            result.append(predict)
            recommend = getRecommendation(predict)
            output+='<td><font size="" color="black">'+comment+'</td><td><font size="" color="black">'+str(predict)+'</td><td><font size="" color="black">'+str(recommend)+'</td></tr>'
        output+= "</table></br>"
        unique, count = np.unique(np.asarray(result), return_counts=True)
        plt.pie(count,labels=unique,autopct='%1.1f%%')
        plt.title('Sentiment Prediction Graph')
        plt.axis('equal')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)      

def TrainCNN(request):
    if request.method == 'GET':
        global mse_error
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">RMSE Error</th></tr>'
        output+='<td><font size="" color="black">CNN+LSTM Algorithm</td><td><font size="" color="black">'+str(mse_error)+'</td></tr>'
        output+= "</table></br></br></br>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)
    

def LoadDatasetAction(request):
    if request.method == 'POST':
        global dataset
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("SentimentApp/static/"+fname):
            os.remove("SentimentApp/static/"+fname)
        with open("SentimentApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        datasets = pd.read_csv("SentimentApp/static/"+fname,nrows=1000)
        datasets.fillna(0, inplace = True)
        columns = datasets.columns
        datasets = datasets.values
        output='<table border=1 align=center width=100%><tr>'
        for i in range(len(columns)):
            output += '<th><font size="" color="black">'+columns[i]+'</th>'
        output += '</tr>'
        for i in range(0, 300):
            output += '<tr>'
            for j in range(len(columns)):
                output += '<td><font size="" color="black">'+str(datasets[i,j])+'</td>'
            output += '</tr>'
        output+= "</table></br></br></br></br>"
        #print(output)
        context= {'data':output}
        return render(request, 'UserScreen.html', context)    

def LoadDataset(request):
    if request.method == 'GET':
       return render(request, 'LoadDataset.html', {})  

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})

def SignupAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        
        status = 'none'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'sentimentapp',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from signup where username = '"+username+"'")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == email:
                    status = 'Given Username already exists'
                    break
        if status == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'sentimentapp',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO signup(username,password,contact_no,email_id,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = 'Signup Process Completed'
        context= {'data':status}
        return render(request, 'Signup.html', context)

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        option = 0
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'sentimentapp',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password:
                    uname = username
                    option = 1
                    break
        if option == 1:
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'Invalid login details'}
            return render(request, 'UserLogin.html', context)

