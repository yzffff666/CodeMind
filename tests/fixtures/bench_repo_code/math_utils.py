def clamp(value, lower, upper):
    """Keep value inside the inclusive [lower, upper] range."""
    if value < lower:
        return lower
    if value > upper:
        return lower
    return value
