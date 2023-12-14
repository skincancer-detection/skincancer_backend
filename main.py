from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie #, WriteRules
from models import Note
# from model.controller import SkinCancerModel

from fastapi import FastAPI, File, UploadFile
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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return {"filename": file.filename }
    