"""
Aetherveil Sentinel Main API

FastAPI application providing REST endpoints for all security modules.
Includes authentication, rate limiting, and comprehensive API documentation.

Security Level: DEFENSIVE_ONLY
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
import uvicorn

from ..config.config import AetherVeilConfig
from ..modules import get_module, list_modules, ModuleStatus
from . import API_CONFIG, APIVersion, APIStatus
from .v1.reconnaissance import router as recon_router
from .v1.scanning import router as scan_router
from .v1.exploitation import router as exploit_router
from .v1.stealth import router as stealth_router
from .v1.osint import router as osint_router
from .v1.orchestrator import router as orchestrator_router
from .v1.reporting import router as reporting_router
from .middleware import RateLimitMiddleware, SecurityMiddleware, AuditMiddleware

logger = logging.getLogger(__name__)

# Pydantic models for API
class APIInfo(BaseModel):
    """API information model"""
    title: str
    description: str
    version: str
    current_version: str
    status: str
    security_disclaimer: str
    uptime: float
    modules_loaded: List[str]

class HealthCheck(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    uptime: float
    modules: Dict[str, str]
    api_version: str

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    message: str
    timestamp: datetime
    request_id: Optional[str] = None

class ModuleStatus(BaseModel):
    """Module status model"""
    module: str
    status: str
    version: str
    last_activity: Optional[str] = None

# Global variables
app_start_time = None
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global app_start_time
    app_start_time = time.time()
    
    # Startup
    logger.info("Starting Aetherveil Sentinel API...")
    
    # Initialize configuration
    config = AetherVeilConfig()
    app.state.config = config
    
    # Initialize modules
    await initialize_modules(config)
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Aetherveil Sentinel API...")
    await shutdown_modules()
    logger.info("API shutdown complete")

async def initialize_modules(config: AetherVeilConfig):
    """Initialize all security modules"""
    try:
        # Import and initialize modules
        from ..modules.reconnaissance import create_reconnaissance_module
        from ..modules.scanning import create_scanning_module
        from ..modules.exploitation import create_exploitation_module
        from ..modules.stealth import create_stealth_module
        from ..modules.osint import create_osint_module
        from ..modules.orchestrator import create_orchestrator_module
        from ..modules.reporting import create_reporting_module
        
        # Create module instances
        recon_module = create_reconnaissance_module(config)
        scan_module = create_scanning_module(config)
        exploit_module = create_exploitation_module(config)
        stealth_module = create_stealth_module(config)
        osint_module = create_osint_module(config)
        orchestrator_module = create_orchestrator_module(config)
        reporting_module = create_reporting_module(config)
        
        # Start modules
        modules = [
            recon_module, scan_module, exploit_module, stealth_module,
            osint_module, orchestrator_module, reporting_module
        ]
        
        for module in modules:
            await module.start()
            logger.info(f"Started module: {module.module_type.value}")
        
        logger.info("All modules initialized successfully")
        
    except Exception as e:
        logger.error(f"Module initialization failed: {e}")
        raise

async def shutdown_modules():
    """Shutdown all security modules"""
    try:
        module_names = list_modules()
        for module_name in module_names:
            module = get_module(module_name)
            if module:
                await module.stop()
                logger.info(f"Stopped module: {module_name}")
        
        logger.info("All modules shutdown successfully")
        
    except Exception as e:
        logger.error(f"Module shutdown failed: {e}")

# Create FastAPI application
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Add custom middleware
app.add_middleware(AuditMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(RateLimitMiddleware, calls_per_minute=60)

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Authenticate API requests"""
    # In production, implement proper JWT token validation
    # For now, we'll use a simple token check
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Simple token validation (replace with proper implementation)
    if credentials.credentials != "aetherveil-api-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {"user": "api_user", "permissions": ["read", "write"]}

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            message=exc.detail,
            timestamp=datetime.utcnow(),
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred",
            timestamp=datetime.utcnow(),
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

# Root endpoints
@app.get("/", response_model=APIInfo)
async def root():
    """Get API information"""
    global app_start_time
    uptime = time.time() - app_start_time if app_start_time else 0
    
    return APIInfo(
        title=API_CONFIG["title"],
        description=API_CONFIG["description"],
        version=API_CONFIG["version"],
        current_version=API_CONFIG["current_version"].value,
        status=API_CONFIG["status"].value,
        security_disclaimer=API_CONFIG["security_disclaimer"],
        uptime=uptime,
        modules_loaded=list_modules()
    )

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    global app_start_time
    uptime = time.time() - app_start_time if app_start_time else 0
    
    # Check module health
    modules_health = {}
    for module_name in list_modules():
        module = get_module(module_name)
        if module:
            modules_health[module_name] = module.status.value
        else:
            modules_health[module_name] = "unknown"
    
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow(),
        uptime=uptime,
        modules=modules_health,
        api_version=API_CONFIG["current_version"].value
    )

@app.get("/modules", response_model=List[ModuleStatus])
async def get_modules_status(current_user: dict = Depends(get_current_user)):
    """Get status of all modules"""
    modules = []
    
    for module_name in list_modules():
        module = get_module(module_name)
        if module:
            status_info = await module.get_status()
            modules.append(ModuleStatus(
                module=module_name,
                status=module.status.value,
                version=getattr(module, 'version', 'unknown'),
                last_activity=status_info.get('last_activity')
            ))
    
    return modules

@app.post("/modules/{module_name}/start")
async def start_module(module_name: str, current_user: dict = Depends(get_current_user)):
    """Start a specific module"""
    module = get_module(module_name)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module {module_name} not found")
    
    try:
        success = await module.start()
        if success:
            return {"message": f"Module {module_name} started successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to start module {module_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/modules/{module_name}/stop")
async def stop_module(module_name: str, current_user: dict = Depends(get_current_user)):
    """Stop a specific module"""
    module = get_module(module_name)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module {module_name} not found")
    
    try:
        success = await module.stop()
        if success:
            return {"message": f"Module {module_name} stopped successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to stop module {module_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/modules/{module_name}/status")
async def get_module_status(module_name: str, current_user: dict = Depends(get_current_user)):
    """Get detailed status of a specific module"""
    module = get_module(module_name)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module {module_name} not found")
    
    try:
        status_info = await module.get_status()
        return status_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include module routers
app.include_router(recon_router, prefix="/api/v1/reconnaissance", tags=["reconnaissance"])
app.include_router(scan_router, prefix="/api/v1/scanning", tags=["scanning"])
app.include_router(exploit_router, prefix="/api/v1/exploitation", tags=["exploitation"])
app.include_router(stealth_router, prefix="/api/v1/stealth", tags=["stealth"])
app.include_router(osint_router, prefix="/api/v1/osint", tags=["osint"])
app.include_router(orchestrator_router, prefix="/api/v1/orchestrator", tags=["orchestrator"])
app.include_router(reporting_router, prefix="/api/v1/reporting", tags=["reporting"])

# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=API_CONFIG["title"],
        version=API_CONFIG["version"],
        description=API_CONFIG["description"],
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer"
        }
    }
    
    # Add security to all endpoints
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if method != "options":
                openapi_schema["paths"][path][method]["security"] = [{"bearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )