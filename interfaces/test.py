# Create a simple test file named test_server.py:

import http.server
import socketserver

PORT = 8765

handler = http.server.SimpleHTTPRequestHandler
httpd = socketserver.TCPServer(("0.0.0.0", PORT), handler)
print(f"Server running at 0.0.0.0:{PORT}")
httpd.serve_forever()