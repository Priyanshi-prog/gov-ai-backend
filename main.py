from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from engine import get_ai_recommendation

app = FastAPI()

# This allows your Flutter app to talk to the API without security blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "AI Policy Hub API is running!"}

@app.get("/ask")
async def ask_ai(job: str, state: str, query: str):
    answer = get_ai_recommendation(job, state, query)
    return {"status": "success", "data": answer}
