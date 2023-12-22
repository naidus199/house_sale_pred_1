import uvicorn
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class house_features(BaseModel):
    area: int
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: int
    guestroom: int
    basement: int
    hotwaterheating: int
    airconditioning: int
    parking: int
    prefarea: int
    semi_furnished: int
    unfurnished: int

# Load the pre-trained model
model = joblib.load('linear_regression.pkl')

@app.get('/ping')
async def ping():
    return 'Welcome to my website'

@app.post('/predict')
async def predict_houseprice(features: house_features):
    # Convert the received features to a list for model prediction
    features_list = [features.area, features.bedrooms, features.bathrooms, features.stories,
                     features.mainroad, features.guestroom, features.basement, features.hotwaterheating,
                     features.airconditioning, features.parking, features.prefarea, 
                     features.semi_furnished, features.unfurnished]
    
    # Reshape the features list for prediction
    features_array = [features_list]  # convert to 2D array for prediction
    
    # Make prediction using the loaded model
    prediction = model.predict(features_array)
    
    return f'The predicted house price: {prediction[0]}'

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
