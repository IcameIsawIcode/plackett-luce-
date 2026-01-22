"""Basic usage example of Plackett-Luce model."""

from plackett_luce import PlackettLuce
import numpy as np

# Example: Movie rankings
# 0=Movie A, 1=Movie B, 2=Movie C, 3=Movie D
rankings = [
    [0, 1, 2, 3],  # User 1's ranking
    [0, 2, 1, 3],  # User 2's ranking
    [1, 0, 2, 3],  # User 3's ranking
    [0, 1, 3, 2],  # User 4's ranking
    [0, 2, 3, 1],  # User 5's ranking
]

# Fit the model
print("Fitting Plackett-Luce model...")
model = PlackettLuce(n_items=4, method='mm')
model.fit(rankings)

# Display results
print("\nMovie Strengths:")
movie_names = ['Movie A', 'Movie B', 'Movie C', 'Movie D']
for i, strength in enumerate(model.params):
    print(f"  {movie_names[i]}: {strength:.3f}")

print("\nRanking (best to worst):")
for rank, item_id in enumerate(model.rank_items(), 1):
    print(f"  {rank}. {movie_names[item_id]}")

# Compute probability of a specific ranking
test_ranking = [0, 1, 2, 3]
prob = model.probability(test_ranking)
print(f"\nProbability of ranking {test_ranking}: {prob:.4f}")
