#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request, send_from_directory
from main import *

app = Flask(__name__)

df = read_data()
X_train, X_test, y_train, y_test,dataX,datay=data_preprocessing(read_data())
clff = None
result=None
@app.route('/')
def ss():
    return render_template('login.html')


@app.route('/test_row',methods = ['POST', 'GET'])
def test_row():
   if request.method == 'POST':
      l_var = []
      l_var.append(request.form['gen']) 
      l_var.append(request.form['Age'])
      l_var.append(request.form['smok'])
      l_var.append(request.form['cpd'])
      l_var.append(request.form['BPM'])
      l_var.append(request.form['BS'])
      l_var.append(request.form['HBP'])
      l_var.append(request.form['D'])
      l_var.append(request.form['TC'])    
      l_var.append(request.form['sbp'])
      l_var.append(request.form['dbp'])    
      l_var.append(request.form['BMI'])          
      l_var.append(request.form['HR'])
      l_var.append(request.form['G'])
         
      l_var = list(map(int, l_var))
      l_var = np.array(l_var).reshape(1, -1)
      clff=logistic_regression(X_train, y_train)
      d=row_probability(clff,l_var)
        
      global result
      result=d
      return render_template('test.html',name=result)  


if __name__ == '__main__':
   app.run(debug = True)






