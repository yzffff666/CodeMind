def slugify(text):
    """Normalize words into a lowercase dash-separated slug."""
    return "-".join(text.lower().split(" "))
