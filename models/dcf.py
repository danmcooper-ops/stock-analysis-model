# models/dcf.py
import numpy as np

def calculate_dcf(free_cash_flows, discount_rate, terminal_value, periods):
    discounted_fcf = [fcf / (1 + discount_rate) ** t for t, fcf in enumerate(free_cash_flows, 1)]
    discounted_terminal = terminal_value / (1 + discount_rate) ** periods
    return sum(discounted_fcf) + discounted_terminal
