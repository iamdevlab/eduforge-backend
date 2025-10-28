from dotenv import load_dotenv

load_dotenv()

# --- ADD THESE IMPORTS ---
# Import your database engine, Base, and your User model
# I'm using the path 'app.services.database' from your users.py file
from app.services.database import engine, Base
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes_auth, routes_questions, lesson_plan, routes_subscriptions


# This line tells SQLAlchemy to create all tables (defined by 'Base')
# in your Supabase database if they don't already exist.
Base.metadata.create_all(bind=engine)
# ---------------------

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
app.include_router(routes_subscriptions.router)


@app.get("/")
def read_root():
    return {"message": "EduForge API is running"}
