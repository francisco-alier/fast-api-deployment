# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union

class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list]
    item_id: int

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def greetings():
    return {"greeting": "Welcome to this amazing app!"}

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/items/")
async def create_item(item: TaggedItem):
    return item