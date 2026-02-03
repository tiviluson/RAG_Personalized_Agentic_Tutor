# RAG Personalized Agentic Tutor

A Canvas-integrated AI tutoring platform that provides personalized learning support through multi-agent RAG system.

## Overview

This is an intelligent tutoring system that:

- Answers questions about course content using RAG
- Provides personalized guidance based on student learning profiles
- Integrates with Canvas LMS for grades, assignments, and course data
- Uses Socratic method for teaching
- Supports document uploads for course materials and student notes

## Tech Stack

- **Agents**: CrewAI (multi-agent orchestration)
- **Backend**: FastAPI
- **Frontend**: Gradio
- **Databases**: PostgreSQL + pgvector, Qdrant, Redis
- **LLM**: Google Gemini 2.5 (with OpenAI/Anthropic fallback)
- **Integration**: Canvas LMS

## Architecture

5 specialized agents work together:
- **Router Agent**: Classifies student queries
- **Tutor Agent**: Orchestrates responses
- **Course Logistics Agent**: Fetches Canvas data (grades, assignments, due dates, etc.)
- **Student Profile Agent**: Manages learning profiles
- **Course Knowledge Agent**: RAG retrieval from course content

![Architecture Diagram](assets/System_Architecture.png)
