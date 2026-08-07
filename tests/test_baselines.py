from ptcg_ai.agents.baselines import GreedyAgent, RandomAgent
from ptcg_ai.api import Agent, Budget

BUDGET = Budget(seconds=1.0)


def test_agents_satisfy_the_protocol():
    assert isinstance(RandomAgent(seed=0), Agent)
    assert isinstance(GreedyAgent(score=lambda o, a: 0.0, seed=0), Agent)


def test_random_agent_is_deterministic_under_a_seed():
    legal = list(range(20))
    a = [RandomAgent(seed=7).act(None, legal, BUDGET) for _ in range(5)]
    b = [RandomAgent(seed=7).act(None, legal, BUDGET) for _ in range(5)]
    assert a == b


def test_random_agent_only_returns_legal_actions():
    legal = ["x", "y", "z"]
    agent = RandomAgent(seed=1)
    assert all(agent.act(None, legal, BUDGET) in legal for _ in range(50))


def test_greedy_agent_picks_the_highest_scoring_action():
    legal = [1, 5, 3]
    agent = GreedyAgent(score=lambda obs, action: action, seed=0)
    assert agent.act(None, legal, BUDGET) == 5


def test_greedy_agent_breaks_ties_across_the_whole_tied_set():
    legal = [1, 2, 3]
    agent = GreedyAgent(score=lambda obs, action: 0.0, seed=0)
    seen = {agent.act(None, legal, BUDGET) for _ in range(100)}
    assert seen == {1, 2, 3}


def test_budget_safety_margin_shrinks_the_deadline():
    assert Budget(seconds=10.0).with_safety_margin(0.8).seconds == 8.0
