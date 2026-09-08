"""Explicit built-in family registrations; no directory discovery."""

from ..family_registry import register_family
from .layered import Layered
from .structured import Structured

register_family(Layered())

register_family(Structured())
