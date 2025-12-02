"""
Health Status Aggregator Package

A Clean Architecture implementation for health status analysis and aggregation.
"""

from app.container import Container, create_container
from app.main import create_app

__all__ = ["Container", "create_container", "create_app"]
