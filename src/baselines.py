"""
baselines.py
============
Two naive recertification policies our trigger is compared against:

  1. fixed_interval(t, interval)  -- audit every `interval` rounds,
     regardless of observed drift. Standard current practice
     ("recertify annually", etc.)

  2. always_audit(t)              -- audit every single round (the
     "safe but expensive" upper bound on cost/accuracy).

Both are evaluated against the SAME ground-truth violation trace as our
trigger, so we can compare (a) number of expensive cryptographic audits
performed and (b) detection lag: how many rounds elapse between the
TRUE fairness violation occurring and the policy actually catching it.
"""


def fixed_interval_policy(t, interval):
    return (t % interval) == 0


def always_audit_policy(t):
    return True


def never_audit_policy(t):
    return False
