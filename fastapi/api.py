from typing import Optional

from fast import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    num = 100
    return {f"Hello": f"World {num}"}
name = 'vishnu'
print(name.title())

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = 'hello'):
    return {"item_id": item_id, "q": q}

def get_name_with_age(name: str, age: int):
    name_with_age = name + " is this old: " + age
    return name_with_age