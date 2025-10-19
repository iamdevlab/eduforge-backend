# EduForge AI Lesson Plan Generator

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

The EduForge AI Lesson Plan Generator is a powerful backend service designed to automate the creation of detailed, curriculum-aligned lesson plans for Nigerian educators. By leveraging generative AI, it produces comprehensive, multi-week educational content tailored to specific subjects, class levels, and terms, significantly reducing the administrative burden on teachers and ensuring high-quality, consistent instructional materials.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Built With](#built-with)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **AI-Powered Content Generation**: Utilizes Google's Gemini models to create rich, detailed lesson plans, including objectives, activities, assessments, and comprehensive summaries.
- **Curriculum-Aligned**: Designed to follow the Nigerian National Curriculum, fetching topics from a ministry service or using user-provided lists.
- **Robust & Resilient**: Implements automatic retries with exponential backoff for AI API calls to handle transient network or service issues.
- **Graceful Fallbacks**: Provides a sensible, structured "skeleton" lesson plan if AI generation fails after all retries, ensuring service availability.
- **Concurrent & Performant**: Uses `asyncio` to generate multiple weeks of a lesson plan concurrently, with configurable rate limiting and concurrency controls to manage API costs and avoid rate limits.
- **Data Validation**: Employs Pydantic for strict, type-safe validation of all incoming requests and AI-generated data.
- **Configurable**: Key parameters like API keys, model names, and concurrency limits are easily configured via environment variables.
- **Intelligent Caching**: Caches ministry scheme-of-work data to reduce redundant API calls and improve performance.

---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing.

### Prerequisites

You need to have Python 3.9+ and a package manager like `pip` installed.

### Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your_username/eduforge-backend.git
    ```
2.  **Navigate to the project directory**
    ```bash
    cd eduforge-backend
    ```
3.  **Install Python dependencies** (assuming a `requirements.txt` file)
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up environment variables**
    Create a `.env` file in the root of the project and add your Google AI API key.

    ```dotenv
    # .env
    AI_API_KEY="YOUR_GOOGLE_AI_API_KEY"
    ```

---

## Usage

The primary function is `generate_lesson_plan`. You can import and call it from within the application, for example, in a FastAPI route.

Here is a basic example of how to use the service in a separate Python script:

```python
import asyncio
from datetime import date
from app.services.ai_lesson_plan_generator import generate_lesson_plan

async def main():
    try:
        print("Generating lesson plan...")
        lesson_plan = await generate_lesson_plan(
            school_name="Future Leaders Academy",
            state="Lagos",
            lga="Ikeja",
            subject="Basic Science",
            class_level="Basic 7 (Jss 1)",
            term="First Term",
            resumption_date=date(2024, 9, 9),
            duration_weeks=3
        )
        print("Lesson Plan Generated Successfully!")
        print(f"School: {lesson_plan.school_name}")
        print(f"Subject: {lesson_plan.subject}")
        print(f"Number of Weeks: {len(lesson_plan.weeks)}")
        # print(lesson_plan.model_dump_json(indent=2))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Running the tests

Explain how to run the automated tests for this system.

### Unit Tests

Explain how to run unit tests.

```bash
npm test
```

---

## Deployment

Add additional notes about how to deploy this on a live system.

---

## Built With

*   React - The web framework used
*   Node.js - Backend runtime environment
*   MongoDB - Database

---

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

---

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

---

## Acknowledgments

*   Hat tip to anyone whose code was used
*   Inspiration
*   etc.