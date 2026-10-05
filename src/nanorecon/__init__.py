"""NanoRecon command line application.

compress: POD5 -> intermediate token file (.nrpod); decompress: token file -> POD5.
The heavy runtime (JAX/Flax) is imported only by the business commands.
"""

__version__ = "1.0.0"
APP_NAME = "nanorecon"
