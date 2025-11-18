# from dotenv import load_dotenv

# load_dotenv()

# from app.services.database import engine, Base
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from app.api import routes_auth, routes_questions, lesson_plan, routes_subscriptions

# # Create database tables
# Base.metadata.create_all(bind=engine)

# app = FastAPI(title="EduForge Backend")

# # Enhanced CORS settings
# origins = [
#     "http://localhost:8000",
#     "http://localhost:5173",
#     "http://localhost:3000",
#     "http://127.0.0.1:3000",
#     "http://127.0.0.1:5173",
#     "https://www.eduforgeplanner.com",
#     "https://eduforge-frontend.pages.dev",
#     "https://eduforgeplanner.com",
#     "https://api.eduforgeplanner.com",
#     "https://eduforge-backend-181749081267.us-central1.run.app",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
#     allow_headers=[
#         "Content-Type",
#         "Authorization",
#         "Access-Control-Allow-Origin",
#         "Access-Control-Allow-Headers",
#         "Access-Control-Allow-Methods",
#         "Access-Control-Allow-Credentials",
#     ],
#     expose_headers=["*"],
#     max_age=600,
# )


# # Add CORS preflight handler
# @app.options("/{rest_of_path:path}")
# async def preflight_handler(rest_of_path: str) -> dict:
#     return {}


# # Add specific CORS headers middleware
# @app.middleware("http")
# async def add_cors_headers(request, call_next):
#     response = await call_next(request)
#     if request.method == "OPTIONS":
#         response.headers["Access-Control-Allow-Origin"] = ", ".join(origins)
#         response.headers["Access-Control-Allow-Methods"] = (
#             "GET, POST, PUT, DELETE, OPTIONS, PATCH"
#         )
#         response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
#         response.headers["Access-Control-Allow-Credentials"] = "true"
#     return response


# # Routers
# app.include_router(routes_auth.router)
# app.include_router(routes_questions.router, prefix="/questions", tags=["questions"])
# app.include_router(lesson_plan.router, prefix="/api", tags=["lesson_plan"])
# app.include_router(routes_subscriptions.router)


# @app.get("/")
# def read_root():
#     return {"message": "EduForge API is running"}


# @app.get("/health")
# def health_check():
#     return {"status": "healthy"}  # from dotenv import load_dotenv

from dotenv import load_dotenv

load_dotenv()


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
    "https://www.eduforgeplanner.com",
    "https://eduforge-frontend.pages.dev",
    "https://eduforgeplanner.com",
    
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
