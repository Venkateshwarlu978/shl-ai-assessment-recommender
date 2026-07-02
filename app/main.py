"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import router
from app.utils.logging import configure_logging
from app.utils.settings import get_settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(title=settings.app_name)
    app.include_router(router)
    return app


app = create_app()
