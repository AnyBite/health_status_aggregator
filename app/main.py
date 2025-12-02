"""
Health Status Aggregator Application

This module bootstraps the FastAPI application with Clean Architecture.
"""

from fastapi import FastAPI

from app.container import create_container
from app.presentation import router, set_container


def create_app() -> FastAPI:
    """
    Application factory function.
    
    Creates and configures the FastAPI application with all dependencies
    properly wired using the Clean Architecture pattern.
    """
    # Create the DI container
    container = create_container()
    
    # Set the container for dependency injection
    set_container(container)
    
    # Create FastAPI app
    app = FastAPI(
        title="Health Status Aggregator",
        description="AI-powered health status analysis and aggregation service",
        version="1.0.0",
    )
    
    # Include the API router
    app.include_router(router)
    
    return app


# Create the app instance for uvicorn
app = create_app()
