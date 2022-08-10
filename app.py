import numpy as np
from flask import Flask, request, jsonify, render_template
#from flask_ngrok import run_with_ngrok
import pickle


app = Flask(__name__)

#run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():

    ten = float(request.args.get('ten'))
    twell = float(request.args.get('twell'))
    btech = float(request.args.get('btech'))
    seven = float(request.args.get('seven'))
    six = float(request.args.get('six'))
    five = float(request.args.get('five'))
    final = float(request.args.get('final'))
    med = float(request.args.get('med'))
    model1=float(request.args.get('model1'))
    if model1==0:
        model=pickle.load(open('PlacementAnalysis_RF.pkl','rb'))
        accr="76.66%"
    elif model1==1:
        model=pickle.load(open('PlacementAnalysis_KNN.pkl','rb'))
        accr="63.33%"
    elif model1==2:
        model=pickle.load(open('PlacementAnalysis_DT.pkl','rb'))
        accr="70.00%"
    elif model1==3:
        model=pickle.load(open('PlacementAnalysis_SVM_linear.pkl','rb'))
        accr="76.67%"
    elif model1==4:
        model=pickle.load(open('PlacementAnalysis_SVM_RBF.pkl','rb'))
        accr="63.33%"
    elif model1==5:
        model=pickle.load(open('PlacementAnalysis_SVM_Sigmoid.pkl','rb'))
        accr="63.33%"
    elif model1==6:
        model=pickle.load(open('PlacementAnalysis_NB.pkl','rb'))
        accr="73.33%"

    prediction = model.predict([[ten, twell, btech, seven, six, five, final, med]])
    if prediction==1:
         message="Booyah! you will be placed"
    else:
        message="Unfortunately, you will not be placed, better luck next time"

    return render_template('index.html', prediction_text=message, accuracy_text='Accuracy of Model :{}'.format(accr))

if __name__ == "__main__":
    app.run(debug=True)
