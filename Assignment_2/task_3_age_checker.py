def is_adult(age):
    """Return True if age is 18 or above. Raise ValueError for negative ages."""
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age >= 18