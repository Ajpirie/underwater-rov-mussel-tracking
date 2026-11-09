"""ROV control integration points. No hardware commands are implemented yet."""
from __future__ import annotations


def CommandROVToTarget(centroid: tuple[float, float, float]) -> bool:
    """Original Runtime.py command placeholder; returns False until integrated."""
    print(f'[ROV placeholder] Commanding ROV to move towards centroid at: {centroid}')
    return False


def commandROVToCollectMussels(centroid: tuple[float, float, float]) -> bool:
    """Original Runtime.py collection placeholder; returns False until integrated."""
    print(f'[ROV placeholder] Commanding ROV to collect mussels at centroid: {centroid}')
    return False
