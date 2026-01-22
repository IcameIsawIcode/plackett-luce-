# Plackett-Luce Model

[![Tests](https://github.com/yourusername/plackett-luce/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/plackett-luce/actions/workflows/tests.yml)

Implementation of the Plackett-Luce model for ranking data.

## Installation
```bash
pip install -e .
```

## Usage
```python
from plackett_luce import PlackettLuce

# Your ranking data (item indices)
rankings = [
    [0, 1, 2],  # Item 0 first, 1 second, 2 third
    [0, 2, 1],
    [1, 0, 2],
]

# Fit model
model = PlackettLuce(n_items=3)
model.fit(rankings)

# Get item strengths
print("Item strengths:", model.params)

# Get ranking
print("Items ranked by strength:", model.rank_items())

# Get probability of a ranking
prob = model.probability([0, 1, 2])
print(f"P(ranking): {prob}")
```

## Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src/plackett_luce
```

## License

MIT
