from phevaluator import evaluate_cards
import random

SUITS = ['d', 's', 'c', 'h']  # diamonds, spades, clubs, hearts
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
DECK_ALL = [r + s for r in RANKS for s in SUITS]

def simulate(hand, table, opponents):
    # Make local copies
    hand = hand[:]
    table = [c for c in table if c]  # strip empty strings if any

    # Build a fresh deck and remove known cards
    used = set(hand) | set(table)
    deck = [c for c in DECK_ALL if c not in used]
    random.shuffle(deck)

    # Deal opponents (2 cards each)
    opp_hands = [[deck.pop(), deck.pop()] for _ in range(opponents)]

    # Complete the board to 5
    while len(table) < 5:
        table.append(deck.pop())

    # Evaluate hero
    my_rank = evaluate_cards(*(table + hand))  # 7 cards exactly

    # Evaluate best opponent
    best_opp = min(
        evaluate_cards(*(table + oh))
        for oh in opp_hands
    )

    # Return 0=WIN, 1=LOSE, 2=SPLIT
    if best_opp < my_rank:
        return 1
    if best_opp == my_rank:
        return 2
    return 0

def monte_carlo(hand, table, opponents=1, samples=100_000):
    """
    Monte Carlo simulation:
    :param hand: list like ['Ad','Qc']
    :param table: list of 0..5 community cards, e.g. ['7s','Ks'] or []
    :param opponents: number of opposing players
    :param samples: number of trials
    :return: [win_pct, lose_pct, tie_pct]
    """
    counts = [0, 0, 0]
    for _ in range(samples):
        outcome = simulate(hand, table, opponents)
        counts[outcome] += 1
    total = float(samples)
    return [c / total for c in counts]

if __name__ == '__main__':
    # Example: hero 7c Qc, full board known, 3 opponents
    print(monte_carlo(['7c', 'Qc'], ['7s','Ks','9c','6c','Kc'], opponents=3, samples=100_000))
