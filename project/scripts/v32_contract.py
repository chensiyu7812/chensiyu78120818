#!/usr/bin/env python3
"""Compatibility shim for the installed V3.2 contract module.

The implementation lives in ``metacom_pm.v32_contract`` so clean pytest and
external callers do not depend on the repository's current working directory.
"""

from metacom_pm.v32_contract import *  # noqa: F401,F403
