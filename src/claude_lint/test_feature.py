"""Test feature with violations."""
import subprocess

# Violation 1: Using emojis (guideline says never use emojis)
WELCOME_MESSAGE = "Welcome to the feature! 🎉 Let's get started! ✨"


# Violation 2: Using a CLASS (guideline says NEVER USE CLASSES)
class DataProcessor:
    def __init__(self, data):
        self.data = data

    def process(self):
        return self.data.upper()


# Violation 3: Missing type hints (guideline says use type hints throughout)
def process_data(input_data):
    return input_data.strip()


# Violation 4: Function over 50 lines (guideline says keep under 50 lines)
# Violation 5: Missing docstring (guideline says write comprehensive docstrings)
def very_long_function(a, b, c, d, e, f, g):
    result = 0

    if a > 0:
        result += a

    if b > 0:
        result += b

    if c > 0:
        result += c

    if d > 0:
        result += d

    if e > 0:
        result += e

    if f > 0:
        result += f

    if g > 0:
        result += g

    # Some more processing
    temp = result * 2
    temp2 = temp + 10
    temp3 = temp2 - 5
    temp4 = temp3 * 3
    temp5 = temp4 / 2
    temp6 = temp5 + 100
    temp7 = temp6 - 50
    temp8 = temp7 * 4
    temp9 = temp8 / 3
    temp10 = temp9 + 200

    final = temp10
    return final


# Violation 6: Documentation mentioning pip install instead of uv
def install_dependencies():
    """
    Install dependencies.

    Run: pip install -r requirements.txt
    """
    subprocess.run(["pip", "install", "package"], check=True)
