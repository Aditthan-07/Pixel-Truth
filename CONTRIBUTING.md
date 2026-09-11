# Contributing to PixelTruth

Thank you for your interest in contributing to **PixelTruth**! We welcome bug reports, feature enhancements, documentation improvements, and pull requests.

## Development Setup

1. **Fork and Clone the Repository**
   `ash
   git clone https://github.com/Aditthan-07/Pixel-Truth.git
   cd Pixel-Truth
   `

2. **Create and Activate a Virtual Environment**
   `ash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   `

3. **Install Dependencies**
   `ash
   pip install -r requirements.txt
   pip install pytest
   `

4. **Run the Test Suite**
   `ash
   python -m unittest discover -s tests
   `

## Pull Request Guidelines

- Ensure all existing unit tests pass before submitting your PR.
- Add unit tests for new functionality or bug fixes under the 	ests/ directory.
- Follow PEP 8 style conventions for Python code.
- Write clear, concise commit messages.
