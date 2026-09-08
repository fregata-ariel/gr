"""Explicit built-in family registrations; no directory discovery."""

from ..family_registry import register_family
from .layered import Layered

register_family(Layered())
