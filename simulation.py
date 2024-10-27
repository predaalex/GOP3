from phevaluator import evaluate_cards
import random
import time

suits = ['d', 's', 'c', 'h']  # diamonds, spades, clubs, hearths
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
cards = []

start = time.time()
for r in ranks:
    for s in suits:
        cards.append(r + s)


def simulate(hand, table, players):
    hands = []
    deck = random.sample(cards, len(cards))  # shuffle the deck
    hand = hand[:]
    table = table[:]

    full = table + hand
    deck = list(filter(lambda x: x not in full, deck))

    # deal cards to players
    for i in range(players):
        hn = [deck.pop(0), deck.pop(0)]
        hands.append(hn)

    # flop, turn, river
    while len(table) < 5:
        card = deck.pop(0)
        table.append(card)
        full.append(card)
    my_hand_rank = evaluate_cards(full[0], full[1], full[2], full[3], full[4], full[5], full[6])

    for check_hand in hands:
        all_cards = table + check_hand
        opponent = evaluate_cards(all_cards[0], all_cards[1], all_cards[2], all_cards[3], all_cards[4], all_cards[5],
                                  all_cards[6])
        # from the definition of the library we use for hand evaluation, larger evaluations correspond to less strong hands
        # so, the game is won by the player with the smallest hand evaluation
        if opponent < my_hand_rank:
            return 1  # 'LOSE'
        if opponent == my_hand_rank:
            return 2  # 'SPLIT'
        return 0  # 'WIN'


def monte_carlo(hand, table, players=2, samples=50000):
    """
    Monte Carlo simulation
    :param hand: player cards
    :param table: community cards
    :param players: number of opponents
    :param samples: number of simulations
    :return: percentages of win | lose | tie
    """

    result = [0, 0, 0]

    for i in range(samples):
        outcome = simulate(hand, table, players)
        result[outcome] += 1
    return list(map(lambda x: x / samples, result))


if __name__ == '__main__':
    print(monte_carlo(['8h', '5d'], []))
