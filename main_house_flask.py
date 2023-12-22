import pickle


import numpy as np
from flask import Flask,request,render_template
app =Flask(__name__)

model = pickle.load(open("linear_regression.pkl","rb"))

@app.route("/")
def homepage():
	return render_template("index.html")
	

# @app.route("/predict",methods=['POST'])
# def predict():
#     pre_sal = lr.predict([[int(x) for x in request.form.values()]])
#     return render_template("index.html",prediction_text = "your Salary is "+str(pre_sal[0]))
    
#@app.route("/app.py", methods = ["GET",  "POST"])
@app.route("/predict", methods = ["GET", "POST"])
def predict():
    if request.method == "POST":
        area = int(request.form.get('area'))
        bedrooms = int(request.form.get('bedrooms'))
        bathrooms = int(request.form.get('bathrooms'))
        stories = int(request.form.get('stories'))
        mainroad = int(request.form.get('mainroad'))
        guestroom = int(request.form.get('guestroom'))
        basement = int(request.form.get('basement') )
        hotwaterheating = int(request.form.get('hotwaterheating'))
        airconditioning = int(request.form.get('airconditioning'))
        parking = int(request.form.get('parking'))
        prefarea = int(request.form.get('prefarea'))
        semi_furnished = int(request.form.get('semi_furnished'))
        unfurnished = int(request.form.get('unfurnished'))

        all_features = [area, bedrooms, bathrooms, stories, mainroad, guestroom,
       basement, hotwaterheating, airconditioning, parking, prefarea,
       semi_furnished, unfurnished]

    predict_houseprice = model.predict([all_features])
    return render_template("index.html",prediction_text = "The House price is "+str(predict_houseprice))
    
    
    
if __name__ == "__main__":
    app.run(port=5000,debug=True)