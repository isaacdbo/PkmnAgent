import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath("__file__")), "cg-lib"))
sys.path.append(os.path.join(os.getcwd(), "cg-lib"))

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import math
import multiprocessing
import random
import time

import diag
from ablation import rewards as ablation_rewards
import torch
import torch.nn
import torch.nn.functional
import torch.optim

# Measured (2026-08-09, this codebase's model + batch-1 inference): 8 threads
# is ~15% SLOWER than 1 for this model's forward pass (small dims, batch=1 —
# thread-coordination overhead exceeds any benefit from splitting tiny matmuls
# across cores). Set at import time, so every process gets it — the main
# process and every spawned self-play worker (each re-imports this module).
torch.set_num_threads(1)

from cg.api import (
    AreaType,
    Card,
    CardType,
    LogType,
    Observation,
    OptionType,
    PlayerState,
    Pokemon,
    SearchState,
    SelectContext,
    all_attack,
    all_card_data,
    search_begin,
    search_end,
    search_step,
    to_observation_class,
    State
)

from cg.game import battle_start, battle_finish, battle_select

# Load all card data from the API's helper function
all_card = all_card_data()
# Create a lookup table (dictionary) to quickly access card data by its cardId
card_table = {c.cardId:c for c in all_card}
card_count = max(all_card, key=lambda c: c.cardId).cardId + 1 # Max Card ID + 1

attack_count = max(all_attack(), key=lambda a: a.attackId).attackId + 1 # Max Attack ID + 1

num_words_encoder = 24
# Per-pokemon word: 46 + 4*card_count; 4 pokemon words + 2 player words + 5 misc words
encoder_size = 300 + 22 * card_count

# Domain-knowledge card IDs resolved from card_table at load time
_TR_PROTON_ID    = next((cid for cid, d in card_table.items() if d.name == "Team Rocket's Proton"), 1220)
_ULTRA_BALL_ID   = next((cid for cid, d in card_table.items() if d.name == "Ultra Ball"), 1121)
_GRASS_ENERGY_ID = next((cid for cid, d in card_table.items() if d.name == "Basic {G} Energy"), 1)
_SPIDOPS_ID      = 401  # Spidops ex — ability 'Bug Catching Set' fetches energy from deck
_MEWTWO_IDS      = frozenset(cid for cid, d in card_table.items() if "mewtwo" in d.name.lower())
_MIMIKYU_IDS     = frozenset(cid for cid, d in card_table.items() if "mimikyu" in d.name.lower())
_TR_ENERGY_IDS   = frozenset(cid for cid, d in card_table.items()
                              if "rocket" in d.name.lower() and d.cardType == CardType.SPECIAL_ENERGY)

decoder_main_feature = 8 # Feature count of SelectContext.Main
decoder_attack_offset = 14 # First index of Attack feature
decoder_card_offset = decoder_attack_offset + attack_count # First index of Card Feature
decoder_size = decoder_card_offset + (1 + decoder_main_feature + SelectContext.RECOVER_SPECIAL_CONDITION) * card_count # Decoder input vocabulary size

FAST_TEST = os.environ.get("FAST_TEST", "1") == "1"  # ~5 min; set FAST_TEST=0 for real runs
DIAG_DUMP_EVERY_GAMES = 25

# Reward ablation arm, selected with REWARD_SPEC=<name> (default "baseline",
# which is exactly the behaviour this file had before the harness existed).
# Read at import time because self-play fans out through 'spawn' workers that
# re-import this module in a fresh interpreter — the environment is what
# survives that boundary. See ablation/rewards.py.
REWARD_SPEC = ablation_rewards.active_spec()

DECK_DIFF_COEF = float(os.environ.get("DECK_DIFF_COEF", 0.01))
if REWARD_SPEC.board_diff_coef is not None:
    DECK_DIFF_COEF = REWARD_SPEC.board_diff_coef

# Where checkpoints are written and re-loaded. Overridable so an ablation
# sweep can give each arm its own directory; without that, arms started from
# each other's checkpoints and the comparison meant nothing.
CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "checkpoints")
MAX_CHILDREN_PER_NODE = 64
# Inference-only experiment (default 0 = off, identical to prior behaviour):
# when > 0, replaces the +0.916 ATTACH domain-bonus logit with a post-softmax
# floor of this value on every ENERGY_ATTACH child's prior, renormalised.
ATTACH_PRIOR_FLOOR = float(os.environ.get("ATTACH_PRIOR_FLOOR", "0"))
SIMULATIONS_PER_MOVE = int(os.environ.get("SIMULATIONS_PER_MOVE", 5 if FAST_TEST else 20))

# Self-play-only exploration (AlphaZero-style). Both only take effect when
# mcts_agent(..., self_play=True) is passed explicitly; no existing call site
# (eval_panel.py, checkpoint_h2h.py, RLTRM2.py's own evaluate()) passes it, so
# the eval/inference path is structurally unaffected — still pure argmax, no noise.
SELF_PLAY_DIRICHLET_ALPHA = float(os.environ.get("SELF_PLAY_DIRICHLET_ALPHA", "0.3"))
SELF_PLAY_DIRICHLET_EPSILON = float(os.environ.get("SELF_PLAY_DIRICHLET_EPSILON", "0.25"))
# Move-selection temperature schedule: turn <= threshold -> temp=1.0 (sample
# proportional to visit count); turn > threshold -> temp=0.0 (argmax, same as eval).
# Default 30, not 10: TURN_NODE_DIST shows ~85-89% of decisions occur at turn 6+
# and games commonly run 40-60+ turns, so a threshold of 10 would leave most
# decisions on deterministic argmax — the regime that produced the prior collapse.
SELF_PLAY_TEMP_HIGH_TURN = int(os.environ.get("SELF_PLAY_TEMP_HIGH_TURN", "30"))

# Prints once per process (this fires at import time, so every run — training
# or eval — states it in the log with no need to go read source to confirm
# it). self_play=True is not a runtime-detectable flag threaded through the
# call chain; it's hardcoded at the two call sites below, so stated as fact,
# not introspected.
print(
    f"[CONFIG] self-play exploration: SELF_PLAY_DIRICHLET_ALPHA={SELF_PLAY_DIRICHLET_ALPHA} "
    f"SELF_PLAY_DIRICHLET_EPSILON={SELF_PLAY_DIRICHLET_EPSILON} "
    f"SELF_PLAY_TEMP_HIGH_TURN={SELF_PLAY_TEMP_HIGH_TURN} "
    f"self_play=True is hardcoded in _play_one_self_play_game and _play_one_cross_play_game "
    f"(the only two call sites that invoke mcts_agent for self-play generation)",
    flush=True,
)
print(f"[CONFIG] reward: {REWARD_SPEC.describe()} DECK_DIFF_COEF={DECK_DIFF_COEF}", flush=True)

# ~1 hour: 20 | ~2 hours: 30
SEARCH_COUNT = SIMULATIONS_PER_MOVE

diag.configure(
    enabled=True,
    verbose=FAST_TEST,
    dump_every_games=(1 if FAST_TEST else DIAG_DUMP_EVERY_GAMES),
)

# Decoder Layer of MyModel
class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int):
        super(DecoderLayer, self).__init__()

        self.attention = torch.nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = torch.nn.Linear(d_model, d_feedforward)
        self.fc2 = torch.nn.Linear(d_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = self.fc1(res)
        y = torch.nn.functional.relu(y)
        y = self.fc2(y)
        return self.norm2(res + y)

# My Transformer Model
class MyModel(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_feedforward: int,
                 num_layers_encoder: int,
                 num_layers_decoder: int
    ):
        super(MyModel, self).__init__()

        self.d_model = d_model

        self.encoder_bag = torch.nn.EmbeddingBag(encoder_size, d_model, mode="sum")
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, 0)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers_encoder, enable_nested_tensor=False)
        self.encoder_fc = torch.nn.Linear(d_model, 1)
        self.decoder_bag = torch.nn.EmbeddingBag(decoder_size, d_model, mode="sum")
        self.decoder = torch.nn.ModuleList()
        for _ in range(num_layers_decoder):
            self.decoder.append(DecoderLayer(d_model, num_heads, d_feedforward))
        self.decoder_fc = torch.nn.Linear(d_model, 1)

    def forward(self,
                index_encoder: torch.Tensor,
                value_encoder: torch.Tensor,
                offset_encoder: torch.Tensor,
                index_decoder: torch.Tensor,
                value_decoder: torch.Tensor,
                offset_decoder: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.encoder_bag(index_encoder, offset_encoder, value_encoder)
        v = v.reshape(-1, num_words_encoder, self.d_model).transpose(0, 1)
        batch_size = v.size(1)
        encoder_out = self.encoder(v)
        v = self.encoder_fc(encoder_out)
        v = torch.tanh(v.mean(0))

        p = self.decoder_bag(index_decoder, offset_decoder, value_decoder)
        p = p.reshape(batch_size, -1, self.d_model).transpose(0, 1)
        for layer in self.decoder:
            p = layer(p, encoder_out)
        p = self.decoder_fc(p)
        p = p.transpose(0, 1).view(batch_size, -1)
        p = torch.tanh(p)
        return (v, p)

# torch.nn.EmbeddingBag input
class SparseVector:
    index: list[int]
    value: list[float]
    offset: list[int]
    pos: int

    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []
        self.pos = 0

    def add(self, index: int, value: float | int | bool):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(value)

    def add_pos(self, pos: int):
        self.pos += pos

    def add_single(self, value: float | int | bool):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos)
            self.value.append(value)
        self.pos += 1

    def word_start(self):
        self.offset.append(len(self.index))

# Add encoder card feature
def add_card(sv: SparseVector, card: Card | Pokemon | None):
    if card != None:
        sv.add(card.id, 1)
    sv.add_pos(card_count)

# Add encoder cards feature
def add_cards(sv: SparseVector, cards: list[Card] | None, value: float):
    if cards != None:
        for card in cards:
            sv.add(card.id, value)
    sv.add_pos(card_count)

# Add encoder Pokémon feature
# Per-slot size: 46 + 4 * card_count
def add_pokemon(sv: SparseVector, poke: Pokemon | None):
    if poke == None:
        sv.add_single(1)
        sv.add_pos(45 + 4 * card_count)
    else:
        sv.add_single(0)
        sv.add_single(poke.hp / poke.maxHp)
        add_card(sv, poke)
        add_cards(sv, poke.tools, 1.0)
        add_cards(sv, poke.energyCards, 0.5)
        sv.add_single(poke.appearThisTurn)
        for e in poke.energies:
            sv.add(int(e), 1)
        sv.add_pos(12)  # 12 EnergyType values
        add_cards(sv, poke.preEvolution, 1.0)
        data = card_table.get(poke.id)
        if data is not None:
            sv.add_single(data.retreatCost / 4)
            if data.weakness is not None:
                sv.add(int(data.weakness), 1)
            sv.add_pos(12)
            if data.resistance is not None:
                sv.add(int(data.resistance), 1)
            sv.add_pos(12)
            sv.add_single(data.ex)
            sv.add_single(data.megaEx)
            sv.add_single(data.tera)
            sv.add_single(data.basic)
            sv.add_single(data.stage1)
            sv.add_single(data.stage2)
        else:
            sv.add_pos(31)  # retreatCost(1) + weakness(12) + resistance(12) + ex/megaEx/tera/basic/stage1/stage2(6)
        
# Add encoder player feature
def add_player(sv: SparseVector, ps: PlayerState):
    sv.add_single(ps.deckCount / 60)
    sv.add_single(len(ps.discard) / 60)
    sv.add_single(ps.handCount / 8)
    sv.add_single(len(ps.bench) / 5)
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)

    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)

    add_cards(sv, ps.discard, 0.25)

# First word of card name lowercased for each known item-locking Pokémon
_ITEM_LOCK_NAMES = frozenset({'budew', 'frillish', 'jellicent'})

# Every component of the shaped reward, in one place: shaped_reward_terms
# starts from a copy of this, and the ablation registry checks it at import so
# a term added here without a matching entry in ablation/rewards.py fails
# loudly instead of quietly surviving an arm that meant to zero all shaping.
SHAPING_TERM_TEMPLATE: dict[str, float] = {
    "PRIZE_DIFF": 0.0,
    "KO_BONUS": 0.0,
    "ENERGY_DENIAL": 0.0,
    "DISRUPTION": 0.0,
    "DAMAGE_PRESSURE": 0.0,
    "PARALYSIS": 0.0,
    "SLEEP": 0.0,
    "ITEM_LOCK_PENALTY": 0.0,
}
ablation_rewards.validate_terms(SHAPING_TERM_TEMPLATE)


def _prize_taken_reward_hit(obs: Observation, your_index: int) -> bool:
    for log in obs.logs:
        if (
            log.type == LogType.MOVE_CARD
            and log.playerIndex == your_index
            and log.fromArea == AreaType.PRIZE
            and log.toArea == AreaType.HAND
        ):
            return True
    return False


def shaped_reward_terms(obs: Observation, your_index: int) -> dict[str, float]:
    state = obs.current
    your = state.players[your_index]
    opp  = state.players[1 - your_index]
    opp_index = 1 - your_index
    terms: dict[str, float] = dict(SHAPING_TERM_TEMPLATE)

    # Prize differential: positive when you've taken more prizes
    terms["PRIZE_DIFF"] += 0.15 * (len(opp.prize) - len(your.prize)) / 6

    opp_disrupted = False
    for log in obs.logs:
        if log.type == LogType.MOVE_CARD:
            # KO bonus: you take a card from your own prize pile
            if (log.playerIndex == your_index and
                    log.fromArea == AreaType.PRIZE and
                    log.toArea == AreaType.HAND):
                terms["KO_BONUS"] += 0.2

            # Energy denial: opponent's energy card goes to discard
            if (log.playerIndex == opp_index and
                    log.toArea == AreaType.DISCARD and
                    log.cardId is not None):
                data = card_table.get(log.cardId)
                if data and data.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
                    terms["ENERGY_DENIAL"] += 0.05

            # Disruption: opponent's hand shuffled to deck (Iono / Judge style); capped at one bonus per observation
            if (not opp_disrupted and
                    log.playerIndex == opp_index and
                    log.fromArea == AreaType.HAND and
                    log.toArea == AreaType.DECK):
                opp_disrupted = True
                terms["DISRUPTION"] += 0.03

        # Bench/active damage: +0.01 per 10 HP lost by an opponent Pokémon
        elif (log.type == LogType.HP_CHANGE and
              log.playerIndex == opp_index and
              log.value is not None and log.value < 0):
            terms["DAMAGE_PRESSURE"] += 0.001 * (-log.value)

        # Paralysis landing on opponent
        elif (log.type == LogType.PARALYZED and
              log.playerIndex == opp_index and
              log.isRecover is False):
            terms["PARALYSIS"] += 0.03

        # Sleep landing on opponent
        elif (log.type == LogType.ASLEEP and
              log.playerIndex == opp_index and
              log.isRecover is False):
            terms["SLEEP"] += 0.03

    # Item lock: penalise unplayable items while an item-locking Pokémon is on the opponent's field
    if obs.select and your.hand:
        opp_names = set()
        for p in (opp.active + opp.bench):
            if p is not None:
                data = card_table.get(p.id)
                if data:
                    opp_names.add(data.name.lower().split()[0])
        if opp_names & _ITEM_LOCK_NAMES:
            playable = {o.index for o in obs.select.option if o.type == OptionType.PLAY}
            for i, c in enumerate(your.hand):
                if i not in playable:
                    data = card_table.get(c.id)
                    if data and data.cardType == CardType.ITEM:
                        terms["ITEM_LOCK_PENALTY"] -= 0.02

    # Ablation hook. Under the default "baseline" spec every weight is 1.0 and
    # this returns `terms` unchanged; an arm that zeroes shaping does it here,
    # at the single place every shaping term passes through, rather than by
    # editing the term computations above.
    return REWARD_SPEC.apply_shaping(terms)


def shaped_reward(obs: Observation, your_index: int) -> float:
    return sum(shaped_reward_terms(obs, your_index).values())


def count_evolutions(ps: PlayerState) -> int:
    """Count Stage-1 and Stage-2 Pokémon in play (active + bench)."""
    count = 0
    for p in (ps.active + ps.bench):
        if p is not None:
            d = card_table.get(p.id)
            if d and (d.stage1 or d.stage2):
                count += 1
    return count


def board_reward(state: 'State', your_index: int) -> float:
    """State-based heuristic added to MCTS leaf value only (never to training targets).
    Discourages deck-out strategies from random deck determinisation."""
    you = state.players[your_index]
    opp = state.players[1 - your_index]

    # Penalise running down your own deck relative to opponent's;
    # random determinisation otherwise makes milling look artificially attractive.
    return (opp.deckCount - you.deckCount) * DECK_DIFF_COEF


def extract_revealed_cards(obs: Observation, your_index: int) -> list[int]:
    """Return card IDs the opponent just revealed (visible in this step's logs)."""
    opp_index = 1 - your_index
    visible_areas = {AreaType.DISCARD, AreaType.ACTIVE, AreaType.BENCH, AreaType.STADIUM}
    revealed: list[int] = []
    for log in obs.logs:
        if log.playerIndex != opp_index:
            continue
        if log.type == LogType.MOVE_CARD and log.cardId and log.toArea in visible_areas:
            revealed.append(log.cardId)
        elif log.type == LogType.PLAY and log.cardId:
            revealed.append(log.cardId)
        elif log.type == LogType.EVOLVE and log.cardIdAfter:
            revealed.append(log.cardIdAfter)
    return list(set(revealed))


_BOARD_AFFECTING_LOG_TYPES = {
    LogType.SWITCH,
    LogType.CHANGE,
    LogType.PLAY,
    LogType.ATTACH,
    LogType.EVOLVE,
    LogType.DEVOLVE,
    LogType.MOVE_ATTACHED,
    LogType.ATTACK,
    LogType.HP_CHANGE,
    LogType.POISONED,
    LogType.BURNED,
    LogType.ASLEEP,
    LogType.PARALYZED,
    LogType.CONFUSED,
}


def _option_type(o) -> OptionType:
    try:
        return OptionType(o.type)
    except Exception as exc:
        raise ValueError(f"Unknown OptionType value: {o.type}") from exc


def _log_type(log) -> LogType:
    try:
        return LogType(log.type)
    except Exception as exc:
        raise ValueError(f"Unknown LogType value: {log.type}") from exc


def _extract_result_reason(obs: Observation) -> int | None:
    for log in obs.logs:
        if _log_type(log) == LogType.RESULT:
            return log.reason
    print("[WARN] RESULT log missing at terminal state; reason set to UNKNOWN", flush=True)
    return None


class GameOutcome:
    """How a finished game ended, for reward specs that care.

    Carried alongside the raw result so `_backup_and_store` can compute a
    terminal value that depends on the ending (deck-out vs prizes) or on the
    turn count, without re-deriving either from an observation that the
    parallel workers no longer hold by the time samples are stored.
    """

    __slots__ = ("cause", "final_turn", "reason_code")

    def __init__(self, cause: str | None, final_turn: int | None, reason_code: int | None):
        self.cause = cause
        self.final_turn = final_turn
        self.reason_code = reason_code

    @classmethod
    def from_final_obs(cls, final_obs: Observation) -> 'GameOutcome':
        reason = _extract_result_reason(final_obs)
        return cls(ablation_rewards.cause_from_reason(reason), final_obs.current.turn, reason)


def _diag_step_features(prev_obs: Observation, selected: list[int], next_obs: Observation) -> tuple[bool, bool, bool, bool]:
    if prev_obs.select is None or prev_obs.current is None:
        raise ValueError("Expected selectable observation in _diag_step_features")

    actor = prev_obs.current.yourIndex
    options = prev_obs.select.option
    chosen = [options[i] for i in selected]
    chosen_types = [_option_type(o) for o in chosen]

    attach_legal = any(_option_type(o) == OptionType.ATTACH for o in options)
    attach_made = any(t == OptionType.ATTACH for t in chosen_types)

    board_affecting = any(
        (log.playerIndex == actor and _log_type(log) in _BOARD_AFFECTING_LOG_TYPES)
        for log in next_obs.logs
    )

    shuffle_with_resources = False
    if prev_obs.select.context == SelectContext.MAIN and len(chosen) == 1 and chosen_types[0] == OptionType.PLAY:
        play = chosen[0]
        card = prev_obs.current.players[actor].hand[play.index]
        data = card_table.get(card.id)
        if data and data.cardType == CardType.SUPPORTER:
            playable_hand_indices = {
                o.index for o in options if _option_type(o) == OptionType.PLAY
            }
            playable_cards_remained = any(idx != play.index for idx in playable_hand_indices)
            shuffle_happened = any(
                _log_type(log) == LogType.MOVE_CARD
                and log.fromArea == AreaType.HAND
                and log.toArea == AreaType.DECK
                for log in next_obs.logs
            )
            shuffle_with_resources = playable_cards_remained and shuffle_happened

    return (attach_legal, attach_made, board_affecting, shuffle_with_resources)


# Reserved slots for a future opponent-archetype belief feature — NOT wired in.
# get_encoder_input below writes a uniform 1/6 placeholder into these 6 slots;
# it does not read AgentState.opponent_belief. Order matches the ARCHETYPES
# dict in __main__ and must stay in sync with it if/when this goes live. Do not
# wire real belief values in until every game-playing path tracks belief
# identically (see FINDINGS.md 2026-08-09 "update_belief call-site coverage
# gap") — that flag, and the separate one on update_belief's uniform
# staples-vs-signature-cards likelihood, both still apply.
BELIEF_ARCHETYPE_ORDER = ("dragapult", "grimmsnarl", "lucario", "mega_lopunny", "slop_box", "other")
BELIEF_SLOT_COUNT = len(BELIEF_ARCHETYPE_ORDER)  # 6
BELIEF_PLACEHOLDER = 1.0 / BELIEF_SLOT_COUNT

# Encoder schema footprint, used by the assertion at the end of
# get_encoder_input to catch silent schema drift (a field added/removed
# without updating this check). Mirrors add_pokemon's documented per-slot
# size ("Per-slot size: 46 + 4*card_count") and add_player's per-call size.
_POKEMON_SLOT_SIZE = 46 + 4 * card_count
_PLAYER_SLOT_SIZE = 16 + card_count
_MISC_SCALAR_COUNT = 7  # leading 1 + turn/firstPlayer/supporterPlayed/energyAttached/retreated/turnActionCount


def get_encoder_input(obs: Observation, your_deck: list[int]) -> SparseVector:
    your_index = obs.current.yourIndex
    state = obs.current

    sv = SparseVector()
    for i in range(2):
        ps = state.players[i ^ your_index]
        for j in range(8): # For bench
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                add_pokemon(sv, ps.bench[j])
            else:
                add_pokemon(sv, None)
            if j != 7:  # Not last
                sv.pos = pos  # Return to the previous position
    
    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        if 0 < len(ps.active):
            add_pokemon(sv, ps.active[0])
        else:
            add_pokemon(sv, None)

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        add_player(sv, ps)
        
    sv.word_start()
    add_cards(sv, state.players[your_index].hand, 0.25)
        
    sv.word_start()
    for id in your_deck:
        sv.add(id, 0.25)
    sv.add_pos(card_count)
        
    sv.word_start()
    add_cards(sv, state.stadium, 1.0)

    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10)
    sv.add_single(state.firstPlayer == your_index)
    sv.add_single(state.supporterPlayed)
    sv.add_single(state.energyAttached)
    sv.add_single(state.retreated)
    sv.add_single(state.turnActionCount / 10)

    # Reserved belief slots (see BELIEF_ARCHETYPE_ORDER above) — placeholder
    # only, not real belief. One add_single per archetype, in order.
    for _ in BELIEF_ARCHETYPE_ORDER:
        sv.add_single(BELIEF_PLACEHOLDER)

    expected_pos = (
        2 * _POKEMON_SLOT_SIZE  # bench: one permanent slot per player (the other 7 reuse the same range)
        + 2 * _POKEMON_SLOT_SIZE  # active
        + 2 * _PLAYER_SLOT_SIZE  # player state, x2 players
        + card_count  # your hand
        + card_count  # your own decklist
        + card_count  # stadium
        + _MISC_SCALAR_COUNT
        + BELIEF_SLOT_COUNT
    )
    assert sv.pos == expected_pos, (
        f"encoder schema drift: sv.pos={sv.pos} != expected={expected_pos} — "
        f"a field was added/removed in get_encoder_input without updating this check"
    )
    return sv

def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None

# Add decoder Main Select feature
def decoder_main(sv: SparseVector, feature_index: int, card: Card | Pokemon | None):
    if card != None:
        sv.add(decoder_card_offset + feature_index * card_count + card.id, 1)
        
# Add decoder Card ID feature
def decoder_card_id(sv: SparseVector, context: SelectContext, card_id: int):
    sv.add(decoder_card_offset + (decoder_main_feature + context) * card_count + card_id, 1)

# Add decoder Card feature
def decoder_card(sv: SparseVector, context: SelectContext, card: Card | Pokemon | None):
    if card != None:
        decoder_card_id(sv, context, card.id)

def get_decoder_input(obs: Observation, actions: list[list[int]]) -> SparseVector:
    sv = SparseVector()
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    context = obs.select.context
    for action in actions:
        sv.word_start()
        
        if len(action) == 0:
            sv.add(0, 1)
            continue
        
        for i in action:
            o = obs.select.option[i]
            match o.type:
                case OptionType.END:
                    sv.add(1, 1)
                case OptionType.YES:
                    sv.add(2, 1)
                case OptionType.NO:
                    sv.add(3, 1)
                case OptionType.SPECIAL_CONDITION:
                    sv.add(4 + o.specialConditionType, 1)
                case OptionType.NUMBER:
                    sv.add(9 + min(o.number, 4), 1)
                case OptionType.ATTACK:
                    sv.add(decoder_attack_offset + o.attackId, 1)
                case OptionType.PLAY:
                    decoder_main(sv, 0, ps.hand[o.index])
                case OptionType.ATTACH:
                    decoder_main(sv, 1, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 2, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.EVOLVE:
                    decoder_main(sv, 3, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 4, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.ABILITY:
                    decoder_main(sv, 5, get_card(obs, o.area, o.index, your_index))
                case OptionType.DISCARD:
                    decoder_main(sv, 6, get_card(obs, o.area, o.index, your_index))
                case OptionType.RETREAT:
                    decoder_main(sv, 7, ps.active[0])
                case OptionType.CARD:
                    decoder_card(sv, context, get_card(obs, o.area, o.index, o.playerIndex))
                case OptionType.TOOL_CARD:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    decoder_card(sv, context, card.tools[o.toolIndex])
                case OptionType.ENERGY_CARD | OptionType.ENERGY:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    decoder_card(sv, context, card.energyCards[o.energyIndex])
                case OptionType.SKILL:
                    decoder_card_id(sv, context, o.cardId)

    return sv

# Evaluate with My Model
def eval_nn(sv_enc: SparseVector, sv_dec:SparseVector, model: MyModel) -> tuple[float, list[float]]:
    device = next(model.parameters()).device
    value, policy = model(
        torch.tensor(sv_enc.index, dtype=torch.int32, device=device),
        torch.tensor(sv_enc.value, dtype=torch.float32, device=device),
        torch.tensor(sv_enc.offset, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.index, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.value, dtype=torch.float32, device=device),
        torch.tensor(sv_dec.offset, dtype=torch.int32, device=device))

    return (value.tolist()[0][0], policy.tolist()[0])

# Single Training Sample - Used for MCTS backpropagation and training
class LearnSample:
    def __init__(self, value: float, policy: list[float], sv_enc: SparseVector, sv_dec: SparseVector):
        self.value = value          # value target: z ∈ {+1, 0, -1}, set by _backup_and_store
        self.policy = policy        # policy target: visit-count proportions π(a|s)
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec
        self.action_count = len(policy)
   
# MCTS Node Child
class Child:
    node: 'Node | None'
    select: list[int] # Selected option indices
    prob: float # Probability

    def __init__(self, select: list[int], prob: float):
        self.node = None
        self.select = select
        self.prob = prob

# MCTS Node
class Node:
    value: float # Self value
    total: float # Total value
    visit: int # Visit count
    parent: 'Node | None' # Parent node
    children: list[Child]
    state: SearchState # Search State of this node

    def __init__(self, parent: 'Node | None', state: SearchState):
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children = []
        self.state = state

    # Backpropagation value
    def backprop(self, value: float):
        self.total += value
        self.visit += 1
        if self.parent != None:
            self.parent.backprop(value)

def create_node(parent: Node | None,
                search_state: SearchState,
                your_index: int,
                your_deck: list[int],
                model: MyModel
    ) -> tuple[Node, LearnSample | None]:
    node = Node(parent, search_state)
    obs  = search_state.observation
    state = obs.current

    # --- (A) Terminal node: assign outcome; backprop happens in the MCTS loop, not here ---
    if state.result >= 0:
        node.value = 0.0 if state.result == 2 else (1.0 if state.result == your_index else -1.0)
        return (node, None)

    diag.record_node_turn(turn=state.turn)

    # --- Build legal-action list ---
    # MAIN: non-terminal first so ATTACK/END are always present even if > 64 total options
    if obs.select.context == SelectContext.MAIN and obs.select.maxCount == 1:
        _t = (OptionType.ATTACK, OptionType.END)
        non_terminal = [i for i, o in enumerate(obs.select.option) if o.type not in _t]
        terminal     = [i for i, o in enumerate(obs.select.option) if o.type in _t]
        actions = [[i] for i in non_terminal[:MAX_CHILDREN_PER_NODE - len(terminal)]] + [[i] for i in terminal]
    else:
        actions = []
        indices = list(range(obs.select.maxCount))
        for _ in range(MAX_CHILDREN_PER_NODE):
            actions.append(indices.copy())
            for i in range(len(indices)):
                k = len(indices) - i - 1
                if indices[k] < len(obs.select.option) - i - 1:
                    indices[k] += 1
                    for j in range(k + 1, len(indices)):
                        indices[j] = indices[j - 1] + 1
                    break
            else:
                break

    kept_option_indices = {idx for action in actions for idx in action}
    generated_lengths = {len(action) for action in actions}
    try:
        ctx_name = SelectContext(obs.select.context).name
    except Exception as exc:
        raise ValueError(f"Unknown SelectContext value in create_node: {obs.select.context}") from exc

    if obs.select.maxCount > 1:
        diag.record_multiselect(
            context=ctx_name,
            min_count=obs.select.minCount,
            max_count=obs.select.maxCount,
            generated_lengths=generated_lengths,
        )

    option_type_names: list[str] = []
    for o in obs.select.option:
        try:
            option_type_names.append(OptionType(o.type).name)
        except Exception as exc:
            raise ValueError(f"Unknown OptionType value in create_node: {o.type}") from exc
    diag.record_truncation(
        turn=state.turn,
        context=("MAIN" if (obs.select.context == SelectContext.MAIN and obs.select.maxCount == 1) else "COMBO"),
        option_type_names=option_type_names,
        kept_option_indices=kept_option_indices,
        cap=MAX_CHILDREN_PER_NODE,
    )

    # --- (B) Encode state and run NN ---
    sv_enc = get_encoder_input(obs, your_deck)
    sv_dec = get_decoder_input(obs, actions)
    # eval_nn returns (value_scalar, policy_logits_list); logits are tanh outputs in [-1, 1]
    value_pred, policy_logits = eval_nn(sv_enc, sv_dec, model)

    # Node value: NN prediction + log-based shaped reward + state-based board reward
    # (search only — training targets z remain pure game outcomes)
    # No clamping: unclamped values keep shaped-reward information intact for PUCT.
    # Backprop happens in the MCTS loop, not here, to avoid double-counting.
    v = value_pred if state.yourIndex == your_index else -value_pred
    shaping_terms = shaped_reward_terms(obs, your_index)
    board_term = board_reward(state, your_index)
    shaping_terms["BOARD_DECK_DIFF"] = board_term
    diag.record_shaping_terms(shaping_terms)
    v += sum(shaping_terms.values())
    node.value = v

    # --- Domain prior bonuses: additive logit offsets applied before softmax ---
    # Scale tanh outputs by 3 to widen dynamic range before softmax; without this,
    # softmax over [-1, 1] produces near-uniform distributions and weak action differentiation.
    logits = [x * 4.0 for x in policy_logits]  # copy with scaling; length == len(actions)
    if state.yourIndex == your_index:
        your_ps = state.players[your_index]
        ctx = obs.select.context

        if ctx == SelectContext.MAIN:
            has_ultra_ball = any(c.id == _ULTRA_BALL_ID for c in (your_ps.hand or []))
            for i, action in enumerate(actions):
                if not action: continue
                opt = obs.select.option[action[0]]

                if opt.type == OptionType.PLAY and not state.supporterPlayed:
                    card = your_ps.hand[opt.index]
                    data = card_table.get(card.id)
                    if data and data.cardType == CardType.SUPPORTER:
                        logits[i] += 0.916  # log(2.5): supporter every turn
                        if card.id == _TR_PROTON_ID and state.turn <= 2:
                            logits[i] += 0.405  # log(1.5) extra: TR Proton T1

                elif opt.type == OptionType.ATTACH and ATTACH_PRIOR_FLOOR <= 0:
                    energy = get_card(obs, opt.area, opt.index, your_index)
                    target = get_card(obs, opt.inPlayArea, opt.inPlayIndex, your_index)
                    if energy and target:
                        if energy.id == _GRASS_ENERGY_ID and target.id == _SPIDOPS_ID:
                            logits[i] += 0.916  # Grass → Spidops
                        elif energy.id in _TR_ENERGY_IDS and target.id in (_MEWTWO_IDS | _MIMIKYU_IDS):
                            logits[i] += 0.916  # TR Energy → Mewtwo / Mimikyu

                elif opt.type == OptionType.ABILITY:
                    poke = get_card(obs, opt.area, opt.index, your_index)
                    if poke and poke.id == _SPIDOPS_ID and not poke.energyCards:
                        # extra boost when Ultra Ball is in hand — fetched energy = discard fodder
                        logits[i] += 0.916 if has_ultra_ball else 0.693

        elif ctx == SelectContext.TO_HAND:
            for i, action in enumerate(actions):
                if not action: continue
                opt = obs.select.option[action[0]]
                if opt.type == OptionType.CARD:
                    picked = get_card(obs, opt.area, opt.index, opt.playerIndex)
                    if picked:
                        if picked.id == _TR_PROTON_ID and state.turn <= 2:
                            logits[i] += 0.916
                        elif picked.id == _GRASS_ENERGY_ID:
                            logits[i] += 0.693  # Bug Catching Set energy search

        elif (ctx == SelectContext.DISCARD and obs.select.effect is not None
              and obs.select.effect.id == _ULTRA_BALL_ID):
            for i, action in enumerate(actions):
                if not action: continue
                opt = obs.select.option[action[0]]
                if opt.type == OptionType.CARD:
                    to_discard = get_card(obs, opt.area, opt.index, opt.playerIndex)
                    if to_discard:
                        d = card_table.get(to_discard.id)
                        if d and d.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
                            logits[i] += 0.693  # prefer discarding energy for Ultra Ball

    # --- (C) Softmax → normalised priors; build children (all unexpanded) ---
    # Softmax over legal-action logits only: all priors ∈ (0,1) and sum to 1.
    priors = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=0).tolist()
    if ATTACH_PRIOR_FLOOR > 0 and obs.select.context == SelectContext.MAIN:
        for i, action in enumerate(actions):
            if not action:
                continue
            opt = obs.select.option[action[0]]
            if opt.type == OptionType.ATTACH and priors[i] < ATTACH_PRIOR_FLOOR:
                priors[i] = ATTACH_PRIOR_FLOOR
        total = sum(priors)
        if total > 0:
            priors = [p / total for p in priors]
    for i, action in enumerate(actions):
        node.children.append(Child(action, priors[i]))

    # --- (D) LearnSample: encodings stored now; policy/value filled by mcts_agent ---
    sample = LearnSample(
        value  = 0.0,                    # overwritten by _backup_and_store after game ends
        policy = [0.0] * len(actions),   # overwritten by mcts_agent with visit-count proportions
        sv_enc = sv_enc,
        sv_dec = sv_dec,
    )
    return (node, sample)

# We will perform exploration using MCTS and select actions. At the same time, we will also generate training data.
def mcts_agent(obs_dict: dict, your_deck: list[int], model: MyModel,
              opponent_deck_sample: list[int] | None = None,
              self_play: bool = False) -> tuple[list[int], LearnSample]:
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    active = state.players[1 - your_index].active
    search_state = search_begin(
        obs,
        your_deck=random.sample(your_deck, state.players[your_index].deckCount),
        your_prize=random.sample(your_deck, len(state.players[your_index].prize)),
        # Use belief-sampled deck if available; otherwise fall back to generic Snorlax placeholder
        opponent_deck=opponent_deck_sample if opponent_deck_sample is not None
                      else [1072] * state.players[1 - your_index].deckCount,
        opponent_prize=[1] * len(state.players[1 - your_index].prize),
        opponent_hand=[1] * state.players[1 - your_index].handCount, # Fill with Basic Energy.
        opponent_active=[1072] if len(active) > 0 and active[0] == None else []) # Fill with Snorlax.
    # root_sample holds the ROOT's encodings — this is what we train on.
    # Expansion calls inside the loop must NOT overwrite it.
    root, root_sample = create_node(None, search_state, your_index, your_deck, model)
    root_obs = root.state.observation

    # Measurement only: root-specific branching factor (distinct from
    # TRUNC_OPTCOUNT_HIST/NODE_TURN_TOTAL, which pool root + internal
    # expansion nodes together).
    diag.record_root_branching(turn=state.turn, child_count=len(root.children))

    # Root exploration noise (AlphaZero-style): self-play generation only.
    # Mixed in post-softmax, at the root only — never at internal/expansion nodes.
    if self_play and len(root.children) > 0:
        noise = _sample_dirichlet(SELF_PLAY_DIRICHLET_ALPHA, len(root.children))
        eps = SELF_PLAY_DIRICHLET_EPSILON
        for child, n in zip(root.children, noise):
            child.prob = (1 - eps) * child.prob + eps * n

    # Search
    decision_max_depth = 0
    decision_depth_sum = 0.0
    decision_sims_contributed = 0
    decision_prize_reached = False

    #Search count presents the number of simulations to be performed for each decision. The higher the number, the more accurate the decision will be.
    for _ in range(SEARCH_COUNT):
        current = root
        sim_depth = 0
        sim_prize_reached = False
        sim_prize_reached_depth = None

        while True:
            # PUCT selection + descent/expansion are all inside this loop.
            # current advances toward a leaf each iteration; loop breaks when
            # a node is expanded or a terminal is reached.

            total_visits = sum(c.node.visit for c in current.children if c.node)

            best_child = None
            best_score = -1e9
            c_puct = 1.0

            for child in current.children:
                Q = child.node.total / child.node.visit if child.node else 0.0
                N = child.node.visit if child.node else 0
                # At opponent's nodes they minimise our value — negate Q so PUCT
                # correctly steers them toward the action worst for us.
                if current.state.observation.current.yourIndex != your_index:
                    Q = -Q
                U = c_puct * child.prob * math.sqrt(total_visits + 1) / (1 + N)
                score = Q + U
                if score > best_score:
                    best_score = score
                    best_child = child

            if best_child.node is None:
                # Leaf: expand — discard its sample (train only on root position)
                ss = search_step(current.state.searchId, best_child.select)
                next_node, _ = create_node(current, ss, your_index, your_deck, model)
                best_child.node = next_node
                sim_depth += 1
                if _prize_taken_reward_hit(next_node.state.observation, your_index):
                    sim_prize_reached = True
                    if sim_prize_reached_depth is None:
                        sim_prize_reached_depth = sim_depth
                next_node.backprop(next_node.value)
                break  # simulation done; next for-iteration starts from root

            # Already expanded: descend
            current = best_child.node
            sim_depth += 1
            if _prize_taken_reward_hit(current.state.observation, your_index):
                sim_prize_reached = True
                if sim_prize_reached_depth is None:
                    sim_prize_reached_depth = sim_depth
            if current.state.observation.current.result >= 0:
                current.backprop(current.value)
                break  # terminal; next for-iteration starts from root
            # Non-terminal: continue while loop (deeper descent)

        decision_max_depth = max(decision_max_depth, sim_depth)
        decision_depth_sum += sim_depth
        decision_sims_contributed += 1
        decision_prize_reached = decision_prize_reached or sim_prize_reached
        diag.record_sim_ko(reached=sim_prize_reached, depth=sim_prize_reached_depth)

    diag.record_search_decision(
        max_depth=decision_max_depth,
        mean_depth=(decision_depth_sum / decision_sims_contributed if decision_sims_contributed > 0 else 0.0),
        prize_reward_reached=decision_prize_reached,
        sims_contributed=decision_sims_contributed,
        sims_configured=SEARCH_COUNT,
    )

    # Select the move. Eval/default path: deterministic argmax over visit count
    # (unchanged). Self-play path only: temperature=1.0 sampling proportional to
    # visit count through turn SELF_PLAY_TEMP_HIGH_TURN, then temperature=0.0
    # (argmax, identical to eval) for the rest of the game.
    visit_counts = [(child.node.visit if child.node is not None else 0) for child in root.children]

    # Measurement only: distribution of visits-per-root-child (0 / 1 / 2-4 / 5+).
    # Directly measures whether the policy target has support at this sims
    # setting, rather than inferring it from a branching-factor average.
    diag.record_root_visit_dist(visit_counts)

    if self_play and state.turn <= SELF_PLAY_TEMP_HIGH_TURN and sum(visit_counts) > 0:
        max_child = random.choices(root.children, weights=visit_counts, k=1)[0]
    else:
        max_child = None
        max_visit = -1
        for child in root.children:
            if child.node is not None and child.node.visit > max_visit:
                max_visit = child.node.visit
                max_child = child

    # --- Measurement only: root-level ATTACK / ENERGY_ATTACH / ABILITY availability vs. what was chosen ---
    type_children: dict[OptionType, list[Child]] = {}
    for child in root.children:
        if len(child.select) != 1:
            continue
        opt = root_obs.select.option[child.select[0]]
        try:
            t = OptionType(opt.type)
        except Exception:
            continue
        if t in (OptionType.ATTACK, OptionType.ATTACH, OptionType.ABILITY):
            type_children.setdefault(t, []).append(child)

    def _most_visited(children: list[Child]) -> tuple[Child | None, int]:
        best, best_visit = None, -1
        for c in children:
            v = c.node.visit if c.node else 0
            if v > best_visit:
                best, best_visit = c, v
        return best, best_visit

    chosen_visit = max_child.node.visit if max_child.node else 0
    chosen_q = (max_child.node.total / max_child.node.visit) if max_child.node and max_child.node.visit > 0 else None

    for kind, opt_type in (
        ("ATTACK", OptionType.ATTACK),
        ("ENERGY_ATTACH", OptionType.ATTACH),
        ("ABILITY", OptionType.ABILITY),
    ):
        children = type_children.get(opt_type, [])
        legal = bool(children)
        rep, rep_visit = _most_visited(children) if legal else (None, 0)
        rep_q = (rep.node.total / rep.node.visit) if (rep is not None and rep.node and rep.node.visit > 0) else None
        diag.record_root_option_stats(
            kind=kind,
            legal=legal,
            chosen=legal and (max_child in children),
            option_visit=rep_visit,
            option_q=rep_q,
            chosen_visit=chosen_visit,
            chosen_q=chosen_q,
        )

    # --- Measurement only: prior assigned to each legal ATTACH action, split by
    # whether it hits the domain-bonus special case in the ATTACH branch of the
    # per-option logit-bonus block above (energy.id == _GRASS_ENERGY_ID and
    # target.id == _SPIDOPS_ID, or TR-energy onto Mewtwo/Mimikyu), vs. the
    # prior of whatever action search actually chose at that same node ---
    for child in type_children.get(OptionType.ATTACH, []):
        opt = root_obs.select.option[child.select[0]]
        energy = get_card(root_obs, opt.area, opt.index, your_index)
        target = get_card(root_obs, opt.inPlayArea, opt.inPlayIndex, your_index)
        is_special = bool(
            energy and target and (
                (energy.id == _GRASS_ENERGY_ID and target.id == _SPIDOPS_ID) or
                (energy.id in _TR_ENERGY_IDS and target.id in (_MEWTWO_IDS | _MIMIKYU_IDS))
            )
        )
        diag.record_attach_prior(
            is_special=is_special,
            attach_prior=child.prob,
            chosen_prior=max_child.prob,
        )

    # --- Measurement only: combined "energy development" view. A low
    # ENERGY_ATTACH rate alone can't tell "not developing energy" apart from
    # "routing through Spidops' Charging Up ability instead" (attach a Basic
    # Energy from the discard pile — a distinct OptionType.ABILITY option, not
    # ENERGY_ATTACH). Restricted to Spidops' own ability specifically (not
    # every OptionType.ABILITY), since other Pokemon's abilities in this deck
    # aren't energy-development actions and would inflate this otherwise.
    spidops_ability_children: list[Child] = []
    for c in type_children.get(OptionType.ABILITY, []):
        opt = root_obs.select.option[c.select[0]]
        poke = get_card(root_obs, opt.area, opt.index, your_index)
        if poke is not None and getattr(poke, "id", None) == _SPIDOPS_ID:
            spidops_ability_children.append(c)

    energy_dev_children = type_children.get(OptionType.ATTACH, []) + spidops_ability_children
    diag.record_energy_dev_stats(
        legal=bool(energy_dev_children),
        chosen=bool(energy_dev_children) and (max_child in energy_dev_children),
    )

    # Measurement only: is any Basic {G} Energy already sitting in your own
    # discard pile at this decision? Spidops' ability accepts any Basic
    # Energy, not just grass, so this specifically measures the grass-setup
    # precondition, not general ability legality.
    diag.record_grass_in_discard(
        present=any(c.id == _GRASS_ENERGY_ID for c in (root_obs.current.players[your_index].discard or []))
    )

    # Build policy target from visit count proportions (AlphaZero-style)
    total_visits = sum(visit_counts)
    if total_visits > 0:
        root_sample.policy = [v / total_visits for v in visit_counts]
    else:
        root_sample.policy = [child.prob for child in root.children]  # fallback to priors

    # Measurement only: zero-mass fraction and Shannon entropy of the FULL
    # policy target vector (not the ATTACK/ENERGY_ATTACH-only proxy) — this is
    # the exact quantity the cross-entropy loss trains against.
    _policy_zero_fraction = (
        sum(1 for p in root_sample.policy if p == 0.0) / len(root_sample.policy)
        if root_sample.policy else 0.0
    )
    _policy_entropy = -sum(p * math.log(p) for p in root_sample.policy if p > 0)
    diag.record_policy_target(zero_fraction=_policy_zero_fraction, entropy=_policy_entropy)

    # value will be overwritten by _backup_and_store; set interim MCTS estimate
    root_sample.value = root.total / root.visit

    search_end()
    return (max_child.select, root_sample)


# Helper class to construct batch inputs for the neural network.
class LearnInput:
    index: list[int]
    value: list[float]
    offset: list[int]

    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []

    def add(self, sv: SparseVector):
        count = len(self.index)
        self.index.extend(sv.index)
        self.value.extend(sv.value)
        for o in sv.offset:
            self.offset.append(o + count)

# Opponent for evaluation.
def _sample_dirichlet(alpha: float, k: int) -> list[float]:
    # Symmetric Dirichlet(alpha) via independent Gamma(alpha,1) draws, normalised.
    # Stdlib-only (no numpy import) to match the rest of the file's use of `random`.
    samples = [random.gammavariate(alpha, 1.0) for _ in range(k)]
    total = sum(samples)
    return [s / total for s in samples] if total > 0 else [1.0 / k] * k


def random_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount) # Select at random.

# For displaying progress.
def progress(count: int, text: str):
    current = 0
    while True:
        percent = 100 * current // count
        sys.stderr.write(f"\r{text} {percent}%   ")
        sys.stderr.flush()
        if(current >= count):
            sys.stderr.write("\n")
            sys.stderr.flush()
            break
        yield current
        current += 1

# --- Opponent archetype belief ---
# Module level (not inside __main__): _cross_play_worker below needs these in
# every process, including spawned workers, which import this module fresh
# and never execute the __main__ block (only the process that starts as
# __main__ does). This also matches BELIEF_ARCHETYPE_ORDER above.
dragapult = [119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 112, 112, 305, 66, 235, 140, 1071, 1227, 1227, 1227, 1227,1182, 1182, 1182, 1198, 1198, 1240, 1086, 1086, 1086, 1086, 1152, 1152, 1152, 1152, 1121, 1121, 1121, 1121, 1120, 1120, 1120, 1120, 1097, 1097, 1097, 1213, 1080, 1260, 1260, 5, 5, 5, 5, 2, 2, 2, 7, 7]
grimmsnarl = [646, 646, 646, 646, 647, 647, 647, 648, 648, 648, 112, 112, 112, 112, 860, 860, 860, 104, 104, 104, 235, 235, 689, 1227, 1227, 1227, 1227, 1182, 1182, 1182, 1219, 1219, 1219, 1152, 1152, 1152, 1152, 1086, 1086, 1086, 1097, 1097, 1079, 1122, 1092, 1213, 1174, 1259, 1259, 1259, 1259, 7, 7, 7, 7, 7, 7, 7, 7, 7]
lucario = [677, 677, 677, 678, 678, 678, 676, 676, 676, 673, 673, 674, 674, 675, 675, 1071, 1227, 1227, 1227, 1227, 1213, 1213, 1182, 1182, 1211, 1219, 1229, 1142, 1142, 1142, 1142, 1152, 1152, 1152, 1152, 1121, 1121, 1121, 1121, 1141, 1141, 1141, 1141, 1080, 1174, 1174, 1174, 1252, 1252, 6, 6, 6, 6, 6, 6,6, 6, 6, 6, 6]
mega_lopunny = [305, 305, 305, 65, 66, 66, 66, 848, 848, 848, 849, 849, 849, 109, 791, 174, 869, 1229, 1229, 1229, 1229, 1182, 1182, 1182, 1182, 1225, 1225, 1225, 1227, 1227, 1227, 1121, 1121, 1121, 1121, 1152, 1152, 1152, 1152, 1122, 1122, 1122, 1122, 1086, 1086, 1086, 1174, 1174, 1174, 1264, 1264, 1264, 11, 11, 11, 11, 16, 16, 16, 13]
slop_box = [756, 756, 756, 756, 1071, 1071, 1071, 1071, 272, 272, 272, 272, 184, 184, 184, 108, 108, 140, 140, 791, 209, 979,1198, 1198, 1198, 1198, 1182, 1182, 1182, 1188, 1188, 1205, 1121, 1121, 1121, 1121, 1102, 1102, 1102, 1102, 1146, 1146, 1146, 1088, 1172, 1172, 1250, 1250, 1250, 1250, 5, 5, 5, 5, 3,3, 6, 6, 19, 2]

ARCHETYPES: dict[str, list[int] | None] = {
    "dragapult":    dragapult,
    "grimmsnarl":   grimmsnarl,
    "lucario":      lucario,
    "mega_lopunny": mega_lopunny,
    "slop_box":     slop_box,
    "other":        None,  # unknown / non-meta deck
}
# Cache as frozensets for O(1) membership test
_ARCHETYPE_SETS: dict[str, frozenset[int]] = {
    k: frozenset(v) for k, v in ARCHETYPES.items() if v is not None
}


def initial_belief() -> dict[str, float]:
    n = len(ARCHETYPES)
    return {name: 1.0 / n for name in ARCHETYPES}


def update_belief(belief: dict[str, float], revealed_card_id: int) -> None:
    for name in belief:
        s = _ARCHETYPE_SETS.get(name)
        # "other" gets weak compatibility; known archetypes: 1.0 if in deck, 0.01 if not
        belief[name] *= (0.5 if s is None else (1.0 if revealed_card_id in s else 0.01))
    total = sum(belief.values())
    if total > 0:
        for name in belief:
            belief[name] /= total
    else:
        belief.update(initial_belief())  # reset if all weights collapsed to zero


def sample_opponent_deck_from_belief(belief: dict[str, float], deck_count: int) -> list[int]:
    names = list(belief.keys())
    archetype_name = random.choices(names, weights=[belief[n] for n in names], k=1)[0]
    deck = ARCHETYPES[archetype_name]
    if deck is None or deck_count <= 0:
        return [1072] * deck_count  # Snorlax: always a valid Basic
    sample = random.sample(deck, min(deck_count, len(deck)))
    # search_begin requires at least one Basic Pokémon; ensure one is present
    if not any(card_table.get(c) and card_table[c].basic for c in sample):
        basics = [c for c in deck if card_table.get(c) and card_table[c].basic]
        sample[-1] = random.choice(basics) if basics else 1072
    return sample


def _play_one_self_play_game(deck: list[int], model: 'MyModel') -> tuple[int, list['LearnSample'], list['LearnSample'], 'GameOutcome']:
    """One mirror self-play game: identical logic to run_self_play's inner
    loop body, factored out so both the serial path and the parallel worker
    call the exact same code (not two copies that could silently drift)."""
    obs, _ = battle_start(deck, deck)
    diag.start_game()
    per_player: list[list[LearnSample]] = [[], []]
    while obs["current"]["result"] < 0:
        prev_obs = obs
        yi = obs["current"]["yourIndex"]
        selected, sample = mcts_agent(obs, deck, model, self_play=True)
        per_player[yi].append(sample)
        obs = battle_select(selected)

        prev_obs_obj = to_observation_class(prev_obs)
        next_obs_obj = to_observation_class(obs)
        attach_legal, attach_made, board_affecting, shuffle_with_resources = _diag_step_features(
            prev_obs_obj, selected, next_obs_obj
        )
        diag.record_turn_step(
            turn=prev_obs_obj.current.turn,
            player_index=yi,
            attach_legal=attach_legal,
            attach_made=attach_made,
            board_affecting=board_affecting,
            shuffle_with_resources=shuffle_with_resources,
        )
    battle_finish()
    final_obs_obj = to_observation_class(obs)
    diag.record_game_result(
        final_turn=final_obs_obj.current.turn,
        reason_code=_extract_result_reason(final_obs_obj),
        did_draw=(obs["current"]["result"] == 2),
    )
    diag.record_true_result(result=obs["current"]["result"])
    diag.end_game()
    result = obs["current"]["result"]
    return result, per_player[0], per_player[1], GameOutcome.from_final_obs(final_obs_obj)


def _self_play_worker(
    deck: list[int],
    state_dict: dict,
    game_indices: list[int],
    base_seed: int,
) -> tuple[list[tuple[int, list, list]], dict, int]:
    """Runs a slice of self-play games in a fresh (spawned) process.

    Must be a module-level function — multiprocessing with the 'spawn' start
    method pickles the target function by reference, and closures/nested
    functions (like the old run_self_play, defined inside __main__) can't be
    pickled. Each game is seeded by its own global index (base_seed + game
    index), not by position in this worker's slice, so the set of games
    played — and the aggregate result — doesn't depend on how many workers
    were used or how the games were split among them.
    """
    torch.set_num_threads(1)
    diag.enable_for_worker()

    model = MyModel(128, 2, 256, 3, 1)
    model.load_state_dict(state_dict)
    model.eval()

    results: list[tuple[int, list, list]] = []
    with torch.inference_mode():
        for gi in game_indices:
            random.seed(base_seed + gi)
            torch.manual_seed(base_seed + gi)
            results.append(_play_one_self_play_game(deck, model))

    window_state, games = diag.take_window_snapshot()
    return results, window_state, games


def _play_one_cross_play_game(
    game_index: int,
    m2_deck: list[int], m2_model: 'MyModel',
    opp_deck: list[int], opp_model: 'MyModel',
) -> dict:
    """One cross-play game: identical logic to run_cross_play's inner loop
    body, factored out for the same reason as _play_one_self_play_game.
    Returns a dict rather than a positional tuple so the two roles ("m2" vs
    "opp") are named, not order-dependent — role assignment alternates by
    game_index, and results must route back to the right agent's replay
    buffer regardless of which process computed them.
    """
    m2_belief = initial_belief()
    opp_belief = initial_belief()
    beliefs = {"m2": m2_belief, "opp": opp_belief}
    decks = {"m2": m2_deck, "opp": opp_deck}
    models = {"m2": m2_model, "opp": opp_model}

    if game_index % 2 == 0:
        obs, _ = battle_start(m2_deck, opp_deck)
        roles = ["m2", "opp"]
    else:
        obs, _ = battle_start(opp_deck, m2_deck)
        roles = ["opp", "m2"]

    diag.start_game()
    per_player: list[list[LearnSample]] = [[], []]
    while obs["current"]["result"] < 0:
        prev_obs = obs
        yi = obs["current"]["yourIndex"]
        role = roles[yi]

        obs_obj = to_observation_class(obs)
        for card_id in extract_revealed_cards(obs_obj, yi):
            update_belief(beliefs[role], card_id)

        opp_deck_count = obs["current"]["players"][1 - yi]["deckCount"]
        opp_deck_sample = sample_opponent_deck_from_belief(beliefs[role], opp_deck_count)

        selected, sample = mcts_agent(obs, decks[role], models[role], opp_deck_sample, self_play=True)
        per_player[yi].append(sample)
        obs = battle_select(selected)

        prev_obs_obj = obs_obj
        next_obs_obj = to_observation_class(obs)
        attach_legal, attach_made, board_affecting, shuffle_with_resources = _diag_step_features(
            prev_obs_obj, selected, next_obs_obj
        )
        diag.record_turn_step(
            turn=prev_obs_obj.current.turn,
            player_index=yi,
            attach_legal=attach_legal,
            attach_made=attach_made,
            board_affecting=board_affecting,
            shuffle_with_resources=shuffle_with_resources,
        )
    battle_finish()
    final_obs_obj = to_observation_class(obs)
    diag.record_game_result(
        final_turn=final_obs_obj.current.turn,
        reason_code=_extract_result_reason(final_obs_obj),
        did_draw=(obs["current"]["result"] == 2),
    )
    diag.record_true_result(result=obs["current"]["result"])
    diag.end_game()
    result = obs["current"]["result"]
    return {
        "result": result,
        "roles": roles,
        "samples": per_player,
        "outcome": GameOutcome.from_final_obs(final_obs_obj),
    }


def _cross_play_worker(
    m2_deck: list[int], m2_state_dict: dict,
    opp_deck: list[int], opp_state_dict: dict,
    game_indices: list[int],
    base_seed: int,
) -> tuple[list[dict], dict, int]:
    """Cross-play counterpart to _self_play_worker — see its docstring for
    why this must be module-level. Opponent belief is already reset fresh at
    the start of every game in the existing (serial) implementation, so it
    needs no special handling here: each game is independent either way.
    """
    torch.set_num_threads(1)
    diag.enable_for_worker()

    m2_model = MyModel(128, 2, 256, 3, 1)
    m2_model.load_state_dict(m2_state_dict)
    m2_model.eval()
    opp_model = MyModel(128, 2, 256, 3, 1)
    opp_model.load_state_dict(opp_state_dict)
    opp_model.eval()

    results: list[dict] = []
    with torch.inference_mode():
        for gi in game_indices:
            random.seed(base_seed + gi)
            torch.manual_seed(base_seed + gi)
            results.append(_play_one_cross_play_game(gi, m2_deck, m2_model, opp_deck, opp_model))

    window_state, games = diag.take_window_snapshot()
    return results, window_state, games


if __name__ == "__main__":
    # Juno's M2 List
    #sample_deck = [721,721,722,722,722,722,723,723,723,723,1092,1121,1121,1145,1145,1163,1163,1219,1219,1219,1219,1227,1227,1227,1227,1262,1262,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]

    file_path = "M2Deck.xlsx"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    my_deck = pd.read_excel(file_path, header=None).iloc[:, 0].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)
    loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)

    # --- Training Hyperparameters ---
    BATCH_SIZE = 128
    LAMBDA = 0.9
    REPLAY_BUFFER_MAXLEN = 25_000       # ~500 games × ~50 samples/player/game
    # Mix the visit-count policy target with a uniform distribution over legal
    # (non-padding) actions only, weight epsilon. Default 0.0: current
    # behaviour unchanged, policy_targets untouched. See POLICY_TARGET_ZERO_
    # FRACTION_MEAN / POLICY_TARGET_ENTROPY_MEAN in diag dumps to tell whether
    # this is doing anything.
    POLICY_LABEL_SMOOTHING = float(os.environ.get("POLICY_LABEL_SMOOTHING", "0.0"))
    # ~1 hour values shown; 2-hour alternatives in comments
    # All eight are env-overridable so a sweep can hold training size fixed
    # across arms without editing this file. WARMUP_EPOCHS, MAIN_EPOCHS and
    # POLICY_LABEL_SMOOTHING already were; the game counts were not, which
    # meant a driver passing e.g. MAIN_SELF_PLAY_M2_GAMES was silently ignored.
    def _env_int(name: str, default: int) -> int:
        return int(os.environ.get(name, default))

    WARMUP_EPOCHS             = _env_int("WARMUP_EPOCHS", 1 if FAST_TEST else 4)   # 1hr: 2 - 2h: 3
    WARMUP_SELF_PLAY_GAMES    = _env_int("WARMUP_SELF_PLAY_GAMES", 3 if FAST_TEST else 25)  # 1hr: 15 -2h: 25
    MAIN_EPOCHS               = _env_int("MAIN_EPOCHS", 2 if FAST_TEST else 20)   # 1hr: 8 - 2h: 15
    EVAL_EVERY                = _env_int("EVAL_EVERY", 1 if FAST_TEST else 4)   # 1hr: 2  - 2h: 3
    MAIN_SELF_PLAY_M2_GAMES   = _env_int("MAIN_SELF_PLAY_M2_GAMES", 3 if FAST_TEST else 25)  # 1hr: 15 - 2h: 25
    MAIN_SELF_PLAY_OPP_GAMES  = _env_int("MAIN_SELF_PLAY_OPP_GAMES", 2 if FAST_TEST else 15)   # 1hr: 8 - 2h: 15
    MAIN_CROSS_PLAY_GAMES     = _env_int("MAIN_CROSS_PLAY_GAMES", 2 if FAST_TEST else 12)   # 1hr: 2 - 2h: 12
    EVAL_GAMES_PER_MATCHUP    = _env_int("EVAL_GAMES_PER_MATCHUP", 2 if FAST_TEST else 10)   # 1hr: 5 - 2h: 8

    class AgentState:
        def __init__(self, name: str, deck: list[int]):
            self.name = name
            self.deck = deck
            self.model = MyModel(128, 2, 256, 3, 1).to(device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
            self.replay: deque = deque(maxlen=REPLAY_BUFFER_MAXLEN)
            self.opponent_belief: dict[str, float] = initial_belief()

        def save_checkpoint(self):
            folder = os.path.join(CHECKPOINT_ROOT, self.name)
            os.makedirs(folder, exist_ok=True)
            ts = time.strftime("%Y-%m-%d_%H-%M")
            path = os.path.join(folder, f"model_{ts}.pth")
            torch.save(self.model.state_dict(), path)
            print(f"[{self.name}] Saved: {path}")

        def load_latest_checkpoint(self):
            folder = os.path.join(CHECKPOINT_ROOT, self.name)
            if not os.path.exists(folder):
                return
            files = sorted(glob.glob(os.path.join(folder, "model_*.pth")))
            if not files:
                return
            try:
                self.model.load_state_dict(torch.load(files[-1], map_location=device))
                print(f"[{self.name}] Loaded: {files[-1]}")
            except Exception as e:
                print(f"[{self.name}] Checkpoint load failed ({e}), starting fresh.")

    def _backup_and_store(agent, result: int, player_idx: int, samples: list[LearnSample],
                          outcome: 'GameOutcome | None' = None):
        # Convert result (0/1/2) into the long-term value target from this
        # player's perspective. Under the default "baseline" reward spec this
        # is exactly +1 / 0 / -1 as before; other arms may scale it by how the
        # game ended (deck-out vs prizes) and how long it took, which is what
        # `outcome` carries. A missing outcome falls back to +1 / 0 / -1, so a
        # call site that does not supply one cannot silently change training.
        z = ablation_rewards.terminal_value(
            REWARD_SPEC,
            result=result,
            player_index=player_idx,
            cause=outcome.cause if outcome else None,
            final_turn=outcome.final_turn if outcome else None,
        )

        for s in samples:
            s.value = z      # long-term outcome target
            agent.replay.append(s)

    def run_self_play(agent, n_games: int):
        agent.model.eval()
        with torch.inference_mode():
            for _ in progress(n_games, f"[{agent.name}] self-play"):
                obs, _ = battle_start(agent.deck, agent.deck)
                diag.start_game()
                per_player: list[list[LearnSample]] = [[], []]
                while obs["current"]["result"] < 0:
                    prev_obs = obs
                    yi = obs["current"]["yourIndex"]
                    selected, sample = mcts_agent(obs, agent.deck, agent.model, self_play=True)
                    per_player[yi].append(sample)
                    obs = battle_select(selected)

                    prev_obs_obj = to_observation_class(prev_obs)
                    next_obs_obj = to_observation_class(obs)
                    attach_legal, attach_made, board_affecting, shuffle_with_resources = _diag_step_features(
                        prev_obs_obj, selected, next_obs_obj
                    )
                    diag.record_turn_step(
                        turn=prev_obs_obj.current.turn,
                        player_index=yi,
                        attach_legal=attach_legal,
                        attach_made=attach_made,
                        board_affecting=board_affecting,
                        shuffle_with_resources=shuffle_with_resources,
                    )
                battle_finish()
                final_obs_obj = to_observation_class(obs)
                diag.record_game_result(
                    final_turn=final_obs_obj.current.turn,
                    reason_code=_extract_result_reason(final_obs_obj),
                    did_draw=(obs["current"]["result"] == 2),
                )
                diag.record_true_result(result=obs["current"]["result"])
                diag.end_game()
                result = obs["current"]["result"]
                outcome = GameOutcome.from_final_obs(final_obs_obj)
                for pi in range(2):
                    _backup_and_store(agent, result, pi, per_player[pi], outcome)

    def run_cross_play(m2, opp, n_games: int):
        m2.model.eval()
        opp.model.eval()
        with torch.inference_mode():
            for i in progress(n_games, f"[m2 vs {opp.name}]"):
                # Reset opponent belief at start of each game
                m2.opponent_belief  = initial_belief()
                opp.opponent_belief = initial_belief()

                if i % 2 == 0:
                    obs, _ = battle_start(m2.deck, opp.deck)
                    by_player = [m2, opp]
                else:
                    obs, _ = battle_start(opp.deck, m2.deck)
                    by_player = [opp, m2]
                diag.start_game()
                per_player: list[list[LearnSample]] = [[], []]
                while obs["current"]["result"] < 0:
                    prev_obs = obs
                    yi   = obs["current"]["yourIndex"]
                    agent = by_player[yi]

                    # Update belief from cards the opponent just revealed this step
                    obs_obj = to_observation_class(obs)
                    for card_id in extract_revealed_cards(obs_obj, yi):
                        update_belief(agent.opponent_belief, card_id)

                    # Determinise opponent hidden state from current belief
                    opp_deck_count = obs["current"]["players"][1 - yi]["deckCount"]
                    opp_deck_sample = sample_opponent_deck_from_belief(
                        agent.opponent_belief, opp_deck_count
                    )

                    selected, sample = mcts_agent(obs, agent.deck, agent.model, opp_deck_sample, self_play=True)
                    per_player[yi].append(sample)
                    obs = battle_select(selected)

                    prev_obs_obj = obs_obj
                    next_obs_obj = to_observation_class(obs)
                    attach_legal, attach_made, board_affecting, shuffle_with_resources = _diag_step_features(
                        prev_obs_obj, selected, next_obs_obj
                    )
                    diag.record_turn_step(
                        turn=prev_obs_obj.current.turn,
                        player_index=yi,
                        attach_legal=attach_legal,
                        attach_made=attach_made,
                        board_affecting=board_affecting,
                        shuffle_with_resources=shuffle_with_resources,
                    )
                battle_finish()
                final_obs_obj = to_observation_class(obs)
                diag.record_game_result(
                    final_turn=final_obs_obj.current.turn,
                    reason_code=_extract_result_reason(final_obs_obj),
                    did_draw=(obs["current"]["result"] == 2),
                )
                diag.record_true_result(result=obs["current"]["result"])
                diag.end_game()
                result = obs["current"]["result"]
                outcome = GameOutcome.from_final_obs(final_obs_obj)
                for pi in range(2):
                    _backup_and_store(by_player[pi], result, pi, per_player[pi], outcome)

    def _split_game_indices(n_games: int, n_workers: int) -> list[list[int]]:
        """Evenly split [0, n_games) into up to n_workers contiguous slices.
        Indices are global (not renumbered per slice), so which worker a game
        lands on doesn't affect its seed — see _self_play_worker/_cross_play_worker.
        """
        if n_games <= 0:
            return []
        n_workers = max(1, min(n_workers, n_games))
        base, extra = divmod(n_games, n_workers)
        slices, start = [], 0
        for w in range(n_workers):
            size = base + (1 if w < extra else 0)
            slices.append(list(range(start, start + size)))
            start += size
        return slices

    def run_self_play_parallel(agent, n_games: int, pool: ProcessPoolExecutor, n_workers: int, base_seed: int) -> None:
        agent.model.eval()
        state_dict = {k: v.cpu() for k, v in agent.model.state_dict().items()}
        slices = _split_game_indices(n_games, n_workers)
        futures = [pool.submit(_self_play_worker, agent.deck, state_dict, gi, base_seed) for gi in slices]
        for _, fut in zip(progress(len(futures), f"[{agent.name}] self-play ({len(slices)} workers)"), as_completed(futures)):
            results, window_state, games = fut.result()
            for result, samples0, samples1, outcome in results:
                _backup_and_store(agent, result, 0, samples0, outcome)
                _backup_and_store(agent, result, 1, samples1, outcome)
            diag.merge_worker_window(window_state, games)

    def run_cross_play_parallel(m2, opp, n_games: int, pool: ProcessPoolExecutor, n_workers: int, base_seed: int) -> None:
        m2.model.eval()
        opp.model.eval()
        m2_state_dict = {k: v.cpu() for k, v in m2.model.state_dict().items()}
        opp_state_dict = {k: v.cpu() for k, v in opp.model.state_dict().items()}
        slices = _split_game_indices(n_games, n_workers)
        futures = [
            pool.submit(_cross_play_worker, m2.deck, m2_state_dict, opp.deck, opp_state_dict, gi, base_seed)
            for gi in slices
        ]
        agents_by_role = {"m2": m2, "opp": opp}
        for _, fut in zip(progress(len(futures), f"[m2 vs {opp.name}] ({len(slices)} workers)"), as_completed(futures)):
            results, window_state, games = fut.result()
            for game_result in results:
                result = game_result["result"]
                roles = game_result["roles"]
                samples = game_result["samples"]
                outcome = game_result.get("outcome")
                for pi in range(2):
                    _backup_and_store(agents_by_role[roles[pi]], result, pi, samples[pi], outcome)
            diag.merge_worker_window(window_state, games)

    def train_agent(agent, batch_size: int = BATCH_SIZE):
        n = len(agent.replay)
        if n < batch_size:
            print(f"[{agent.name}] Too few samples ({n}), skipping.")
            return

        # One full pass worth of gradient steps, but each batch is a fresh
        # random draw — true SGD breaks temporal correlation between consecutive
        # game states that would otherwise bias the gradient direction.
        num_batches = n // batch_size
        print(f"[{agent.name}] Training ({n} samples, {num_batches} batches)...")
        agent.model.train()

        for _ in range(num_batches):
            # Fresh random sample each step: diverse gradients, no ordering bias
            batch = random.sample(list(agent.replay), batch_size)

            # --- Build sparse encoder / decoder inputs via LearnInput ---
            # EmbeddingBag expects flat index/value/offset tensors, not dense matrices,
            # so we use LearnInput to pack all samples into a single contiguous buffer.
            input_enc = LearnInput()
            input_dec = LearnInput()
            action_counts: list[int] = []
            for s in batch:
                input_enc.add(s.sv_enc)
                input_dec.add(s.sv_dec)
                action_counts.append(s.action_count)
                # Pad decoder words to max_actions so EmbeddingBag produces a
                # uniform (batch × max_actions) output — empty words sum to zero.
            max_actions = max(action_counts)
            # Reprocess decoder offsets with dynamic padding
            input_dec = LearnInput()
            for s in batch:
                input_dec.add(s.sv_dec)
                for _ in range(max_actions - s.action_count):
                    input_dec.offset.append(len(input_dec.index))  # empty word

            # --- Policy targets: visit-count proportions π(a|s) ---
            # Dynamic width (max_actions) avoids wasting compute on the 64 - max_actions
            # columns that no sample in this batch actually uses.
            padded_policy = [
                s.policy + [0.0] * (max_actions - s.action_count) for s in batch
            ]
            policy_targets = torch.tensor(padded_policy, dtype=torch.float32, device=device)

            # --- Value targets: outcome z ∈ {+1, 0, -1} ---
            # Shape (batch, 1) matches the model's value head output.
            value_targets = torch.tensor(
                [s.value for s in batch], dtype=torch.float32, device=device
            ).unsqueeze(1)

            # --- Validity mask: 1 for real actions, 0 for padding ---
            mask = torch.zeros(batch_size, max_actions, device=device)
            for i, count in enumerate(action_counts):
                mask[i, :count] = 1.0

            # --- Optional label smoothing: mix visit-count target with a
            # uniform distribution over legal (non-padding) actions only.
            # A currently-zero-visit action's target goes from exactly 0 to a
            # fixed epsilon/action_count floor, which caps how far the CE loss
            # can push that action's logit down (unlike an exact-zero target,
            # which places no lower bound on it at all). No-op at epsilon=0.
            if POLICY_LABEL_SMOOTHING > 0:
                uniform_masked = mask / mask.sum(dim=-1, keepdim=True).clamp(min=1)
                policy_targets = (1 - POLICY_LABEL_SMOOTHING) * policy_targets + POLICY_LABEL_SMOOTHING * uniform_masked

            # --- Forward pass ---
            # model returns (value_head, policy_head): (batch,1) and (batch, max_actions)
            out_enc, out_dec = agent.model(
                torch.tensor(input_enc.index,  dtype=torch.int32,   device=device),
                torch.tensor(input_enc.value,  dtype=torch.float32, device=device),
                torch.tensor(input_enc.offset, dtype=torch.int32,   device=device),
                torch.tensor(input_dec.index,  dtype=torch.int32,   device=device),
                torch.tensor(input_dec.value,  dtype=torch.float32, device=device),
                torch.tensor(input_dec.offset, dtype=torch.int32,   device=device))

            # --- Value loss: MSE ---
            # z is a fixed discrete signal in [-1, 1]; MSE penalises all errors equally,
            # which prevents the head from ignoring small residuals near the extremes.
            loss_value = torch.nn.functional.mse_loss(out_enc, value_targets)

            # --- Policy loss: cross-entropy against visit-count proportions ---
            # Padding slots are set to -1e9 before log_softmax so the model is never
            # rewarded for assigning probability to actions that don't exist here.
            # This keeps the MCTS prior well-calibrated: once trained, the policy head
            # will concentrate probability on the actions MCTS found most valuable,
            # so future searches need fewer simulations to reach good decisions.
            masked_logits = out_dec + (mask - 1.0) * 1e9
            log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
            # CE = -∑_a π(a)·log q(a); unvisited actions contribute 0 since π=0.
            loss_policy = -(policy_targets * log_probs).sum(-1).mean()

            agent.optimizer.zero_grad()
            (loss_value + loss_policy).backward()
            # Clip gradients: early in training the policy targets are noisy (few visits),
            # so large gradient steps would destabilise both heads simultaneously.
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
            agent.optimizer.step()

        print(f"[{agent.name}] Done ({num_batches} batches).")

    def evaluate(m2, opponents: list, n_games: int):
        print("=== Evaluation ===")
        m2.model.eval()
        with torch.inference_mode():
            for opp in opponents:
                opp.model.eval()
                res = [0, 0, 0]  # win, loss, draw
                for i in progress(n_games, f"Eval m2 vs {opp.name}"):
                    if i % 2 == 0:
                        obs, _ = battle_start(m2.deck, opp.deck)
                        m2_pi = 0
                    else:
                        obs, _ = battle_start(opp.deck, m2.deck)
                        m2_pi = 1
                    while obs["current"]["result"] < 0:
                        yi = obs["current"]["yourIndex"]
                        if yi == m2_pi:
                            selected, _ = mcts_agent(obs, m2.deck, m2.model)
                        else:
                            selected, _ = mcts_agent(obs, opp.deck, opp.model)
                        obs = battle_select(selected)
                    battle_finish()
                    r = obs["current"]["result"]
                    if r == 2:
                        res[2] += 1
                    elif r == m2_pi:
                        res[0] += 1
                    else:
                        res[1] += 1
                total = res[0] + res[1]
                rate = 100 * res[0] // total if total else 0
                print(f"  vs {opp.name:14s}: {rate:3d}%  ({res[0]}W / {res[1]}L / {res[2]}D)")

    # --- Agent Setup ---
    # M2_ONLY=1: skip instantiating the 5 archetype opponents entirely (not
    # just skip using them) — no model/optimizer/replay buffer created for
    # them, no checkpoint ever written for them. opponent_agents falls out to
    # [] automatically, which already makes every `for opp in
    # opponent_agents:` loop below (main-phase opponent self-play, cross-play)
    # a no-op with no further changes needed. Warm-up still runs — it primes
    # the replay buffer past BATCH_SIZE before train_agent will do anything,
    # not an optional step — it just now only has m2 to iterate over.
    M2_ONLY = os.environ.get("M2_ONLY", "0") == "1"
    all_agents = [AgentState("m2", my_deck)]
    if not M2_ONLY:
        all_agents += [
            AgentState("dragapult",    dragapult),
            AgentState("grimmsnarl",   grimmsnarl),
            AgentState("lucario",      lucario),
            AgentState("mega_lopunny", mega_lopunny),
            AgentState("slop_box",     slop_box),
        ]
    m2_agent        = all_agents[0]
    opponent_agents = all_agents[1:]

    if os.environ.get("SKIP_CHECKPOINT_LOAD", "0") != "1":
        for agent in all_agents:
            agent.load_latest_checkpoint()

    total_games = 0
    run_start = time.time()

    # --- Parallel self-play worker pool ---
    # SELF_PLAY_WORKERS=1 (default) still goes through the pool/worker/merge
    # path, just with one worker — this is deliberate, not a fallback special
    # case: it's what "N=1 worker, verify byte-identical to serial" verifies
    # against. 'spawn', not the Linux default 'fork', because forking a
    # process that already holds a live ctypes handle into the compiled cg
    # engine (and torch's own internal state) is a known class of subtle bugs;
    # spawn re-imports everything cleanly in each worker instead.
    SELF_PLAY_WORKERS = int(os.environ.get("SELF_PLAY_WORKERS", "1"))
    SELF_PLAY_BASE_SEED = int(os.environ.get("SELF_PLAY_BASE_SEED", "20260810"))
    _seed_cursor = SELF_PLAY_BASE_SEED
    _mp_context = multiprocessing.get_context("spawn")
    worker_pool = ProcessPoolExecutor(max_workers=SELF_PLAY_WORKERS, mp_context=_mp_context)

    try:
        # === WARM-UP PHASE ===
        print(f"=== Warm-up Phase ({WARMUP_EPOCHS} epochs) ===")
        for epoch in range(WARMUP_EPOCHS):
            print(f"--- Warm-up Epoch {epoch + 1}/{WARMUP_EPOCHS} ---")
            for agent in all_agents:
                diag.set_agent(agent.name)
                run_self_play_parallel(agent, WARMUP_SELF_PLAY_GAMES, worker_pool, SELF_PLAY_WORKERS, _seed_cursor)
                _seed_cursor += WARMUP_SELF_PLAY_GAMES
                total_games += WARMUP_SELF_PLAY_GAMES
                train_agent(agent)
            for agent in all_agents:
                agent.save_checkpoint()

        # === MAIN PHASE ===
        print(f"=== Main Phase ({MAIN_EPOCHS} epochs) ===")
        for epoch in range(MAIN_EPOCHS):
            epoch_start = time.time()
            print(f"--- Main Epoch {epoch + 1}/{MAIN_EPOCHS} ---")

            # Self-play data collection
            diag.set_agent(m2_agent.name)
            run_self_play_parallel(m2_agent, MAIN_SELF_PLAY_M2_GAMES, worker_pool, SELF_PLAY_WORKERS, _seed_cursor)
            _seed_cursor += MAIN_SELF_PLAY_M2_GAMES
            total_games += MAIN_SELF_PLAY_M2_GAMES
            for opp in opponent_agents:
                diag.set_agent(opp.name)
                run_self_play_parallel(opp, MAIN_SELF_PLAY_OPP_GAMES, worker_pool, SELF_PLAY_WORKERS, _seed_cursor)
                _seed_cursor += MAIN_SELF_PLAY_OPP_GAMES
                total_games += MAIN_SELF_PLAY_OPP_GAMES

            # Cross-play: M2 vs each opponent (both sides collect samples)
            for opp in opponent_agents:
                diag.set_agent(f"m2_vs_{opp.name}")
                run_cross_play_parallel(m2_agent, opp, MAIN_CROSS_PLAY_GAMES, worker_pool, SELF_PLAY_WORKERS, _seed_cursor)
                _seed_cursor += MAIN_CROSS_PLAY_GAMES
                total_games += MAIN_CROSS_PLAY_GAMES

            # Train all agents on their replay buffers
            for agent in all_agents:
                train_agent(agent)

            # Checkpoint all agents
            for agent in all_agents:
                agent.save_checkpoint()

            # Periodic evaluation (no training)
            if (epoch + 1) % EVAL_EVERY == 0:
                evaluate(m2_agent, opponent_agents, EVAL_GAMES_PER_MATCHUP)

            elapsed = time.time() - run_start
            epoch_secs = time.time() - epoch_start
            remaining = elapsed / (epoch + 1) * (MAIN_EPOCHS - epoch - 1)
            print(
                f"Games: {total_games} | Epoch: {epoch_secs/60:.1f}m | "
                f"Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m",
                flush=True
            )

            stop_after = os.environ.get("STOP_AFTER_MAIN_EPOCH")
            if stop_after is not None and (epoch + 1) >= int(stop_after):
                print(f"[STOP] Reached STOP_AFTER_MAIN_EPOCH={stop_after}; halting cleanly.", flush=True)
                break
    finally:
        worker_pool.shutdown(wait=True)
