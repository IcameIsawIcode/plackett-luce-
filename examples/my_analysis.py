
from plackett_luce import PlackettLuce
import numpy as np


# ============================================================================
# Example 1: Simple ranking analysis (Modify this with your data)
# ============================================================================

# Define your items (optional - for display)
items = ["Item A", "Item B", "Item C", "Item D"]

# Your ranking data: list of rankings where each ranking is item indices
# Format: [0, 1, 2, 3] means Item A is 1st, Item B is 2nd, etc.
rankings = [
    [0, 1, 2, 3],  # ranking 1
    [0, 1, 2, 3],  # ranking 2
    [0, 2, 1, 3],  # ranking 3
    [1, 0, 2, 3],  # ranking 4
    [0, 1, 3, 2],  # ranking 5
]

print("=" * 60)
print("PLACKETT-LUCE MODEL ANALYSIS")
print("=" * 60)

# Create and fit the model
model = PlackettLuce(n_items=len(items), method='mm')
model.fit(rankings)

# ============================================================================
# Results
# ============================================================================

print("\n1. ITEM STRENGTHS (learned preferences):")
print("-" * 60)
for idx, item in enumerate(items):
    print(f"   {item}: {model.params[idx]:.4f}")

print("\n2. RANKING (best to worst):")
print("-" * 60)
ranked_indices = model.rank_items()
for rank, idx in enumerate(ranked_indices, 1):
    print(f"   {rank}. {items[idx]} (strength: {model.params[idx]:.4f})")

print("\n3. TOP K ITEMS:")
print("-" * 60)
top_2 = model.top_k(2)
print(f"   Top 2 items: {[items[i] for i in top_2]}")

print("\n4. PROBABILITY OF SPECIFIC RANKINGS:")
print("-" * 60)
test_rankings = [
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [1, 0, 2, 3],
]

for ranking in test_rankings:
    prob = model.probability(ranking)
    ranking_str = " > ".join([items[i] for i in ranking])
    print(f"   P({ranking_str}): {prob:.4f}")

print("\n5. PREDICTION: Most likely ranking")
print("-" * 60)
best_ranking = model.rank_items()
best_ranking_str = " > ".join([items[i] for i in best_ranking])
prob_best = model.probability(best_ranking)
print(f"   {best_ranking_str}")
print(f"   Probability: {prob_best:.4f}")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)


# ============================================================================
# Example 2: How to use with YOUR OWN DATA
# ============================================================================

print("\n\nHOW TO USE WITH YOUR OWN DATA:")
print("-" * 60)
print("""
1. Replace the 'items' list with your items:
   items = ["Product A", "Product B", "Product C", ...]

2. Replace 'rankings' with your ranking data:
   - Each ranking should be a list of item indices in order
   - Example: [0, 2, 1, 3] means: Item 0 is 1st, Item 2 is 2nd, etc.
   
3. Adjust n_items to match the number of items:
   model = PlackettLuce(n_items=len(items))

4. Change method if needed:
   method='mm'   (Minorization-Maximization, faster)
   method='mle'  (Maximum Likelihood Estimation, more accurate)

5. Run the script with: python examples/my_analysis.py
""")
