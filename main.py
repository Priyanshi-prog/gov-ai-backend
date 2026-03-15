from fastapi import FastAPI
from engine import get_ai_recommendation

app = FastAPI()

@app.get("/")
def home():
    return {"message": "AI Personal Policy Assistant is LIVE!"}

@app.get("/ask")
def ask_ai(job: str, state: str, query: str, age: str = "N/A", gender: str = "N/A", income: str = "N/A"):
    try:
        response = get_ai_recommendation(job, state, query, age, gender, income)
        return {"status": "success", "data": response}
    except Exception as e:
        return {"status": "error", "message": str(e)}
