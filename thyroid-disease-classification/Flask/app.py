from __future__ import print_function
from train_build import building_model
import pickle
from flask import Response
import numpy as np
from Logs import logger
from flask import Flask,request,render_template,redirect,url_for
from train_build import building_model
import training_validation
import js2py
from training_validation import train_validation
from DBOperations import operations
app=Flask(__name__,template_folder='templates')
import sys






@app.route("/", methods=['POST','GET'])
def home_page():
    return  render_template('index.html')


@app.route("/training",methods=['GET','POST'])
def training_model():
    loger = logger.App_Logger()
    f = open("Prediction_logs/app_logs.txt", 'a+')
    loger.log(f,"training function is started!!")
    try:
        file_path="Datasets/hypothyroid.csv"
        object=building_model.Build(file_path)
        object.make_a_model(file_path)
        loger.log(f,"training done successfully")
        # return redirect("home")
        f.close()
    except Exception as e:
        loger.log(f, "Error Occurred in training "+e)
        f.close()
        return Response ("Error Occurred %s " % e)
    f.close()
    # return Response("Training done successful")
    return redirect("home")

@app.route('/prediction',methods=['POST','GET'])
def prediction_function():
    loger = logger.App_Logger()
    f = open("Prediction_logs/app_logs.txt", 'a+')
    loger.log(f,"prediction function is started !!")
    try:
        obj = operations.Operation_DB()
        db_name = "thyroid_peoject"
        loger=logger.App_Logger()
        loger.log(f,"prediction function is starter!!")
        model=pickle.load(open("Models/logistic_model.pkl",'rb'))
        values=[float(value) for value in request.form.values()]
        final=np.array([[values]])
        result=model.predict(final[0])
        result=int(result)
        print('++=====+++',values, file=sys.stderr)
        values.append(result)
        obj.insert_on_table(db_name,values)
        if result==0:
            var = "Negative Report"
        else:

            if values[1]<0.5:
                var="Hyperthyroid"  

            else :  
                var="Hypothyroid"
            
        # loger.log(f, result)
        loger.log(f, "prediction function is done successfully")
        f.close()
        return render_template("result.html",value=var)

    except Exception as e:
        f.close()
        return Response ("Error Occurred %s " % e)
    f.close()
    return Response ("Prediction Successfully Done")


if __name__=="__main__":
    app.run(debug=True)

