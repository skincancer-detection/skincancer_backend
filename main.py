from ast import Str
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie #, WriteRules
from models import Note
from model.controller import SkinCancerModel

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import shutil
# from contextlib import asynccontextmanager


# async def init():
#     client = AsyncIOMotorClient("mongodb+srv://jaatisbad9:wOsEbKlAD3RvvIIH@nosql.uz5okma.mongodb.net/?retryWrites=true&w=majority")
#     await init_beanie(database=client.db_name, document_models=[Note])
#     print("Database has been connected !")


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     await init()
#     yield


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with the actual origin of your Node.js application
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(payload: dict):
    model = SkinCancerModel('/home/highvich', 'model_fold_')
    prediction = model.predict_for_image(payload['image_local_path'])
    return prediction
    