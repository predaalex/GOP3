import math
from functools import lru_cache

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def binom_pmf(n, k, p):
    # Simple binomial PMF
    if k < 0 or k > n: return 0.0
    if p <= 0: return 1.0 if k == 0 else 0.0
    if p >= 1: return 1.0 if k == n else 0.0
    # nCk
    from math import comb
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def logistic_continue_prob(raise_over_call, a=-2.0, b=-0.0008):
    """
    Probability a single opponent CONTINUES facing a raise R (over-the-call).
    p(R) = sigmoid(a + b * R).
    Defaults:
      - a = -2.0  -> baseline ~0.12 when R = 0 (cold-call-ish)
      - b negative -> larger R => smaller continue prob
    TUNE a/b to your pool; or pass your own function.
    """
    return sigmoid(a + b * float(raise_over_call))

def decide_action_pro(
    # State
    hand, board, opponents, pot_value, call_value,
    # Betting constraints
    min_raise=0, max_raise=0,
    hero_stack=None, opp_stack=None,  # effective stacks in front of you now
    # Equity model
    get_equity=None,    # function k -> (win_prob, tie_prob); if None, uses simulation.monte_carlo
    equity_samples=20000,
    # Opponent response model
    continue_prob_fn=logistic_continue_prob,  # maps R -> per-opponent continue prob
    # Rake
    rake_percent=0.0, rake_cap=None,
    # Your simulation module (for default equity)
    simulation_module=None,
):
    """
    Returns: dict with
      {
        'best_action': "fold"|"check"|"call"|"raise",
        'amount': 0 or raise size (over-the-call),
        'ev': best EV,
        'ev_call': EV(call),
        'details': {... breakdown ...}
      }
    Assumptions:
      - pot_value is the pot BEFORE your action.
      - call_value is what you must put in to call.
      - When you RAISE by R over the call:
          • You invest call_value + R (capped by hero_stack).
          • Each opponent independently continues with prob p(R).
          • If k opponents continue, each invests up to min(R, opp_stack).
      - If ALL fold (k=0), you win the current pot (no rake).
      - If k>=1, the hand “goes to showdown” now (rake applies) and equity vs k is used.
    """

    # ---- sanity / caps ----
    opponents = max(0, int(opponents))
    hero_stack = float('inf') if hero_stack is None else float(max(0, hero_stack))
    opp_stack  = float('inf') if opp_stack  is None else float(max(0, opp_stack))

    # Cap call and raises by stack
    max_affordable_raise = max(0, hero_stack - call_value)
    if max_raise <= 0:
        max_raise = 0
    if min_raise < 0:
        min_raise = 0
    # respect stack cap
    max_raise = min(max_raise, max_affordable_raise)

    # ---- default equity function (uses your simulation) ----
    if get_equity is None:
        if simulation_module is None:
            raise ValueError("Provide simulation_module or a get_equity(k) function.")
        @lru_cache(None)
        def _equity(k):
            # k = number of opponents that continue
            if k <= 0:
                return (1.0, 0.0)  # trivial: if no caller, you win pot uncontested
            win, lose, tie = simulation_module.monte_carlo(hand, board, players=k, samples=equity_samples)
            # return (win, tie)
            return (win, tie)
        get_equity = _equity
    else:
        # cache user-supplied function too
        get_equity = lru_cache(None)(get_equity)

    # ---- EV helpers ----
    def apply_rake(pot_total):
        if rake_percent <= 0:
            return 0.0
        rake = pot_total * rake_percent
        if rake_cap is not None:
            rake = min(rake, rake_cap)
        return rake

    # EV of checking / calling
    def ev_call_option():
        if call_value == 0:
            # Pure check: no money goes in, pot stays; no one folds by assumption.
            # We go to next street/showdown only if that’s your model; here, treat as call of 0 with 1+ opponents “continuing”.
            # Use current opponents count to get equity.
            k = max(1, opponents)  # avoid trivial k=0 on a check
            win, tie = get_equity(k)
            pot_sd = pot_value  # no chips added
            rake = apply_rake(pot_sd)
            # Net: win*(pot - rake) + tie*0.5*(pot - rake) - lose*0
            lose = max(0.0, 1.0 - win - tie)
            return win*(pot_sd - rake) + tie*0.5*(pot_sd - rake) - lose*0.0
        else:
            # Call: you invest call_value; assume no extra cold-callers (keep simple).
            # If you want cold-callers, model with a small p at R=0 and use binomial like the raise branch.
            k = max(1, opponents)
            win, tie = get_equity(k)
            lose = max(0.0, 1.0 - win - tie)
            pot_sd = pot_value + call_value + k*0.0  # only you add chips with a call in this simple branch
            # If you want others to also call current bet, add k*call_value above.
            rake = apply_rake(pot_sd)
            return win*(pot_sd - rake) + tie*0.5*(pot_sd - rake) - lose*call_value

    # EV of a raise by R (over-the-call)
    def ev_raise_option(R):
        R = clamp(R, 0, max_raise)
        if R <= 0:
            return float('-inf')  # not a real raise
        # Opponent per-head continue prob for this R
        p_cont = clamp(continue_prob_fn(R), 0.0, 1.0)

        ev = 0.0
        # distribution of k callers among 'opponents'
        for k in range(0, opponents + 1):
            pk = binom_pmf(opponents, k, p_cont)
            if pk == 0.0:
                continue

            if k == 0:
                # Everyone folds: you win the pot; your bet comes back; no rake (typical).
                ev += pk * (pot_value)
                continue

            # At least one caller: showdown now.
            # Each caller can only call up to opp_stack, you can only invest up to hero_stack
            hero_invest = clamp(call_value + R, 0, hero_stack)
            caller_each = clamp(R, 0, opp_stack)
            total_callers_contrib = k * caller_each

            pot_sd = pot_value + hero_invest + total_callers_contrib
            rake = apply_rake(pot_sd)

            win, tie = get_equity(k)
            lose = max(0.0, 1.0 - win - tie)

            ev_k = win*(pot_sd - rake) + tie*0.5*(pot_sd - rake) - lose*hero_invest
            ev += pk * ev_k

        return ev

    # ---- Evaluate actions ----
    out_details = {}

    ev_call = ev_call_option()
    out_details['call'] = ev_call

    best_action, best_amount, best_ev = ("check" if call_value == 0 else "call", 0, ev_call)

    # Scan raises in steps of min_raise (nonlinear EV due to folds + equity-by-k),
    # include the max_raise as well.
    if min_raise > 0 and max_raise >= min_raise:
        R = min_raise
        seen = set()
        while R <= max_raise + 1e-9:  # numeric fuzz
            Rr = round(R)  # keep integers if your table uses chips
            if Rr not in seen:
                evR = ev_raise_option(Rr)
                out_details[f'raise_{Rr}'] = evR
                if evR > best_ev:
                    best_action, best_amount, best_ev = ("raise", Rr, evR)
                seen.add(Rr)
            R += min_raise

    # Compare to fold if calling costs chips
    if call_value > 0 and best_ev <= 0.0:
        return {
            'best_action': 'fold',
            'amount': 0,
            'ev': 0.0,
            'ev_call': ev_call,
            'details': out_details
        }

    return {
        'best_action': best_action,
        'amount': best_amount,
        'ev': best_ev,
        'ev_call': ev_call,
        'details': out_details
    }
