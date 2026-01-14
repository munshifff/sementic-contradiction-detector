# Semantic Contradiction Detector

A Python project for detecting semantic contradictions in text using `fake_review_analyzer.py`.

## Project Structure

```
semantic-contradiction-detector/
├── data/ # Datasets used by the project
    ├── dataset.txt
├── notebooks/ # notbook used by the project
    ├── SementicContradiction.ipynb
├── src/ # Core source code
    ├── fake_review_analyzer.py # Main script to run detection
    ├── fake_review_app.py # streamlit app
├── requirements.txt
├── Dockerfile
└── README.md
```

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## Setup

1. Clone the repository:

```bash
git clone https://github.com/munshifff/sementic-contradiction-detector.git
cd sementic-contradiction-detector
```

## Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python src/fake_review_analyzer.py
```

Make sure any input data is in the `data/` folder.
The script will process the data and output results accordingly.

## Demo / Online Usage

Try it online via Hugging Face Spaces:

[Fake Review Analyzer on Hugging Face](https://huggingface.co/spaces/Munshifff/Fake_review_analyzz)
