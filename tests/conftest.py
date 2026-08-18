"""Test runtime initialization shared by optional training tests."""

# The Windows PyArrow/PyTorch wheel combination used by the pinned environment
# requires torch's native runtime to initialize first. Keep torch optional.
try:
    import torch  # noqa: F401
except (ImportError, OSError):
    pass
