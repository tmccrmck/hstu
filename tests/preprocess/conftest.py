"""Configure TensorFlow threading to avoid deadlocks on macOS in pytest."""
import os

os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
