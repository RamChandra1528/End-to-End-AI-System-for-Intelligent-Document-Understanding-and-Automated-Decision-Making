"""
Quick start script to run the Document AI API
"""
import uvicorn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    print("=" * 60)
    print("   Document AI System - Starting API Server")
    print("=" * 60)
    print("\nEndpoints will be available at:")
    print("  - API Docs: http://localhost:8000/docs")
    print("  - API Root: http://localhost:8000")
    print("  - Health: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
