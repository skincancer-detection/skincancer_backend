from ast import Str
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie #, WriteRules
from models import Note
from model.controller import SkinCancerModel
from model.resnet_prediction import PredictHelper

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import shutil
import requests

from dotenv import load_dotenv
load_dotenv()


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with the actual origin of your Node.js application
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def download_file(url, save_path):
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
        return True
    return False


@app.post("/predict")
async def predict(payload: dict):
    
    if payload['image_local_path'].startswith("https://"):

        local_path = '/tmp/'+ payload['image_local_path'].split("/")[-1]
        download_file(payload['image_local_path'], local_path)
        payload['image_local_path'] = local_path
    
    if payload["version"] == "v2":
        prediction = PredictHelper().prediction_helper(payload['image_local_path'], payload)
        return prediction.tolist()[0]
    else: # Base model with v1
        model = SkinCancerModel('/home/highvich', 'model_fold_')
        prediction = model.predict_for_image(payload['image_local_path'])
        return prediction
    