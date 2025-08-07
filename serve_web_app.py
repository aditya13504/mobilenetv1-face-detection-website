"""
Simple HTTP server for the Face Detection Web Application
Serves the web app with proper CORS headers for ONNX model loading
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support"""
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        
        # Add headers for ONNX model files
        if self.path.endswith('.onnx'):
            self.send_header('Content-Type', 'application/octet-stream')
        elif self.path.endswith('.wasm'):
            self.send_header('Content-Type', 'application/wasm')
        elif self.path.endswith('.js'):
            self.send_header('Content-Type', 'application/javascript')
        elif self.path.endswith('.json'):
            self.send_header('Content-Type', 'application/json')
        
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight"""
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        """Custom log message format"""
        print(f"[{self.address_string()}] {format % args}")

def main():
    """Start the HTTP server"""
    # Change to web app directory
    web_app_dir = Path(__file__).parent / "web_app"
    
    if not web_app_dir.exists():
        print("Error: web_app directory not found!")
        print("Make sure this script is in the same directory as the web_app folder.")
        sys.exit(1)
    
    os.chdir(web_app_dir)
    print(f"Serving from: {web_app_dir.absolute()}")
    
    # Server configuration
    PORT = 8000
    Handler = CORSHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"\n🚀 Face Detection Web App Server")
            print(f"📍 Server running at: http://localhost:{PORT}")
            print(f"📁 Serving directory: {web_app_dir.name}")
            print(f"🌐 Open your browser and navigate to: http://localhost:{PORT}")
            print(f"\n📋 Instructions:")
            print(f"   1. Open http://localhost:{PORT} in your browser")
            print(f"   2. Allow camera access when prompted")
            print(f"   3. Click 'Start Camera' to begin")
            print(f"   4. Click 'Start Detection' to detect faces")
            print(f"\n⏹️  Press Ctrl+C to stop the server\n")
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Error: Port {PORT} is already in use!")
            print(f"Try a different port or stop the other server using port {PORT}")
        else:
            print(f"❌ Error starting server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
