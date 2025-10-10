from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes_auth, routes_questions, lesson_plan

app = FastAPI(title="EduForge Backend")

# CORS settings
origins = [
"http://localhost:8000",
"http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
"http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
# app.include_router(routes_auth.router, prefix="/auth", tags=["auth"])
app.include_router(routes_auth.router)
app.include_router(routes_questions.router, prefix="/questions", tags=["questions"])
app.include_router(lesson_plan.router, prefix="/api", tags=["lesson_plan"])
@app.get("/")
def read_root():
    return {"message": "EduForge API is running"}
