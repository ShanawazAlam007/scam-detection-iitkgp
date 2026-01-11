# Gunicorn configuration for Render deployment
# Render automatically sets the PORT environment variable

import multiprocessing
import os

# Bind to the port Render provides (required for Render)
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
backlog = 2048

# Worker configuration - optimized for Render's resources
# Render free tier has limited CPU, so we use fewer workers
workers = int(os.getenv('WORKERS', min(multiprocessing.cpu_count() * 2 + 1, 4)))
worker_class = 'sync'
worker_connections = 1000
timeout = 300  # 5 minutes for slow ML predictions
keepalive = 5
max_requests = 1000  # Restart workers after 1000 requests to prevent memory leaks
max_requests_jitter = 50  # Add randomness to prevent all workers restarting at once

# Logging - Render captures stdout/stderr
accesslog = '-'
errorlog = '-'
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'scam_detector'

# Preload app for better memory efficiency
preload_app = True

# Server hooks
def on_starting(server):
    """Called just before master process is initialized."""
    print("🚀 Starting Gunicorn server on Render...")

def when_ready(server):
    """Called just after server is started."""
    print(f"✓ Server ready on {bind} with {workers} workers")

def on_exit(server):
    """Called just before exiting."""
    print("👋 Shutting down Gunicorn server...")
