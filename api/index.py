import sys
import os

# Ensure root directory is on the Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from api.app import app

# Vercel WSGI entry point
# Flask app instance is exported as 'app'
if __name__ == "__main__":
    app.run(debug=True)
