from .download_ERA5 import download_ERA5
from .download_COSMO1 import download_COSMO1

# Patch requests get to avoid connecting for too long to internet

import requests

_original_get = requests.get


def _req_get(*args, **kwargs):
    timeout = kwargs.pop('timeout', 1)
    return _original_get(*args, **kwargs, timeout=timeout)


requests.get = _req_get
