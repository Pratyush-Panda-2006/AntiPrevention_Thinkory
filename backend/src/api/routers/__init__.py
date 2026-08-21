from src.api.routers.health import router as health_router
from src.api.routers.cdse import router as cdse_router
from src.api.routers.detect import router as detect_router
from src.api.routers.models import router as models_router

__all__ = [
    "health_router",
    "cdse_router",
    "detect_router",
    "models_router",
]
