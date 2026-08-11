"""Verification of illegal-action masking, in both places it must hold.

1. Training side (build_policy_targets_and_mask + masked_policy_log_probs,
   the exact functions train_agent calls):
   - padding/illegal slots get ~0 probability after log_softmax,
   - policy targets are a distribution over legal slots only (sum to 1,
     padding stays exactly 0), with and without label smoothing,
   - the loss gradient w.r.t. padding logits is ~0 (the model is never
     pushed toward or away from actions that don't exist).

2. Search side (mcts_agent on a real engine game, self_play=True so the
   Dirichlet root noise + temperature-sampling path is exercised):
   - every selected action is an index into the engine's legal option list,
   - every stored LearnSample policy is a distribution over that decision's
     legal actions (sums to 1 when any child was visited),
   - _sample_dirichlet returns a valid simplex point.

Run inside the amd64 container (libcg.so is Linux x86-64):
    SIMULATIONS_PER_MOVE=3 python test_masking.py
"""
import os
import random

os.environ.setdefault("SIMULATIONS_PER_MOVE", "3")  # keep the real-game test fast

import torch

import RLTRM2 as R

PASS = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS
    if not cond:
        raise AssertionError(f"FAIL: {name} {detail}")
    PASS += 1
    print(f"ok {PASS}: {name}")


def test_training_side():
    device = torch.device("cpu")
    torch.manual_seed(0)

    # Heterogeneous action counts force real padding: widths 3, 5, 1.
    policies = [
        [0.5, 0.25, 0.25],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0],
    ]
    action_counts = [len(p) for p in policies]

    for smoothing in (0.0, 0.05):
        targets, mask = R.build_policy_targets_and_mask(
            policies, action_counts, device, label_smoothing=smoothing
        )
        check(f"mask marks exactly the legal slots (smoothing={smoothing})",
              all(mask[i, :c].sum().item() == c and mask[i, c:].sum().item() == 0
                  for i, c in enumerate(action_counts)))
        check(f"padding targets are exactly 0 (smoothing={smoothing})",
              all(targets[i, c:].abs().sum().item() == 0.0
                  for i, c in enumerate(action_counts)))
        row_sums = targets.sum(dim=-1)
        check(f"targets renormalise to 1 over legal actions (smoothing={smoothing})",
              torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6),
              f"row_sums={row_sums.tolist()}")

    # Zero-visit sample (all-zero policy): target row must stay all-zero at
    # smoothing=0 -> contributes 0 to CE, never pushes probability anywhere.
    z_targets, _ = R.build_policy_targets_and_mask([[0.0, 0.0]], [2], device, 0.0)
    check("zero-visit sample target stays exactly zero (no spurious gradient)",
          z_targets.abs().sum().item() == 0.0)

    # Gradient check on the actual loss form used in train_agent.
    targets, mask = R.build_policy_targets_and_mask(policies, action_counts, device, 0.05)
    out_dec = torch.randn(3, max(action_counts), requires_grad=True)
    log_probs = R.masked_policy_log_probs(out_dec, mask)

    probs = log_probs.exp()
    pad_mass = (probs * (1 - mask)).sum().item()
    check("softmax probability mass on illegal/padding slots ~ 0",
          pad_mass < 1e-8, f"pad_mass={pad_mass}")
    legal_sums = (probs * mask).sum(dim=-1)
    check("softmax renormalises to 1 over legal slots",
          torch.allclose(legal_sums, torch.ones_like(legal_sums), atol=1e-6))

    loss_policy = -(targets * log_probs).sum(-1).mean()
    loss_policy.backward()
    pad_grad = (out_dec.grad * (1 - mask)).abs().max().item()
    check("policy-loss gradient on illegal/padding logits ~ 0",
          pad_grad < 1e-6, f"max_pad_grad={pad_grad}")


def test_dirichlet():
    random.seed(1234)
    for k in (1, 2, 7, 40):
        n = R._sample_dirichlet(0.3, k)
        check(f"_sample_dirichlet(k={k}) is a valid simplex point",
              len(n) == k and abs(sum(n) - 1.0) < 1e-9 and all(x >= 0 for x in n))
    # Convexity: the root mix (1-eps)*prior + eps*noise preserves sum-to-1.
    priors = [0.7, 0.2, 0.1]
    noise = R._sample_dirichlet(0.3, 3)
    eps = R.SELF_PLAY_DIRICHLET_EPSILON
    mixed = [(1 - eps) * p + eps * n for p, n in zip(priors, noise)]
    check("root Dirichlet mix preserves a normalised prior",
          abs(sum(mixed) - 1.0) < 1e-9 and all(x > 0 for x in mixed))


def test_search_side_real_game():
    """One real engine game: our seat runs the full mcts_agent self-play path
    (Dirichlet noise + temperature sampling + visit-count policy extraction);
    the opponent seat plays uniform random. Every decision is checked."""
    import eval_panel as EP

    random.seed(4242)
    torch.manual_seed(4242)
    deck = EP._read_sample_submission_deck()
    model = R.MyModel(128, 2, 256, 3, 1)
    model.eval()

    decisions = 0
    policies_checked = 0
    with torch.inference_mode():
        obs, _ = R.battle_start(deck, deck)
        while obs["current"]["result"] < 0:
            yi = obs["current"]["yourIndex"]
            obs_obj = R.to_observation_class(obs)
            n_options = len(obs_obj.select.option)
            max_count = obs_obj.select.maxCount
            if yi == 0:
                selected, sample = R.mcts_agent(obs, deck, model, self_play=True)
                decisions += 1
                check_quiet = decisions  # count silently; summary asserts below
                assert all(0 <= idx < n_options for idx in selected), (
                    f"illegal option index in {selected} (n_options={n_options})")
                assert len(selected) <= max_count, (
                    f"selected {len(selected)} options, maxCount={max_count}")
                assert len(set(selected)) == len(selected), f"duplicate indices in {selected}"
                if sample is not None:
                    s = sum(sample.policy)
                    assert s == 0.0 or abs(s - 1.0) < 1e-6, (
                        f"policy target not normalised: sum={s}")
                    assert all(p >= 0 for p in sample.policy)
                    policies_checked += 1
            else:
                selected = EP._random_bot_move(obs)
            obs = R.battle_select(selected)
        R.battle_finish()

    check("real game completed with mcts_agent(self_play=True) on every candidate decision",
          decisions > 0, f"decisions={decisions}")
    check("every selected action was a legal engine option (asserted per decision)", True)
    check("every stored policy target was normalised over legal actions",
          policies_checked > 0, f"policies_checked={policies_checked}")


if __name__ == "__main__":
    test_training_side()
    test_dirichlet()
    test_search_side_real_game()
    print(f"\nALL {PASS} CHECKS PASSED")
