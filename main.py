from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie #, WriteRules
from models import Note

from fastapi import FastAPI
from contextlib import asynccontextmanager
# from src.schema import (
#     PersonCreate,
#     PersonRead,
#     Pet as PetSchema,
# )
# from beanie.operators import In


async def init():
    client = AsyncIOMotorClient("mongodb+srv://jaatisbad9:wOsEbKlAD3RvvIIH@nosql.uz5okma.mongodb.net/?retryWrites=true&w=majority")
    await init_beanie(database=client.db_name, document_models=[Note])
    print("Database has been connected !")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init()
    yield


app = FastAPI(lifespan=lifespan)

@app.post("/person")
async def create_person(note: Note):
    print(note)
    t = await note.insert()
    print(t, t.dict())
    reviews = await Note.find_all().to_list()
    return reviews



# @app.post("/person", response_model=PersonRead)
# async def create_person(person_create: PersonCreate):
#     person = Person(**person_create.model_dump(mode="json"))

#     await person.insert(link_rule=WriteRules.WRITE)
#     return person


# @app.get("/person", response_model=list[PersonRead])
# async def get_person(name: str = None):
#     if name:
#         return await Person.find(Person.name == name).to_list()
#     return await Person.find_all().sort(+Person.name).to_list()


# @app.get("/pet", response_model=list[PetSchema])
# async def get_pet(name: str = None):
#     if name:
#         return await Pet.find(Pet.name == name).to_list()
#     return await Pet.find_all().sort(+Pet.name).to_list()


# @app.post("/pet", response_model=PetSchema)
# async def create_pet(pet_create: PetSchema):
#     pet = Pet(**pet_create.model_dump(mode="json"))

#     await pet.insert()
#     return pet