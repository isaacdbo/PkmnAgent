import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath("__file__")), "cg-lib"))
sys.path.append(os.path.join(os.getcwd(), "cg-lib"))

from collections import deque
import glob
import math
import random
import time

import torch
import torch.nn
import torch.nn.functional
import torch.optim

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

decoder_main_feature = 8 # Feature count of SelectContext.Main
decoder_attack_offset = 14 # First index of Attack feature
decoder_card_offset = decoder_attack_offset + attack_count # First index of Card Feature
decoder_size = decoder_card_offset + (1 + decoder_main_feature + SelectContext.RECOVER_SPECIAL_CONDITION) * card_count # Decoder input vocabulary size

FAST_TEST = True  # flip to False for real training runs

SEARCH_COUNT = 5 if FAST_TEST else 50  # MCTS simulations per move

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

def shaped_reward(obs: Observation, your_index: int) -> float:
    state = obs.current
    your = state.players[your_index]
    opp  = state.players[1 - your_index]
    opp_index = 1 - your_index
    reward = 0.0

    # Prize differential: positive when you've taken more prizes
    reward += 0.15 * (len(opp.prize) - len(your.prize)) / 6

    opp_disrupted = False
    for log in obs.logs:
        if log.type == LogType.MOVE_CARD:
            # KO bonus: you take a card from your own prize pile
            if (log.playerIndex == your_index and
                    log.fromArea == AreaType.PRIZE and
                    log.toArea == AreaType.HAND):
                reward += 0.2

            # Energy denial: opponent's energy card goes to discard
            if (log.playerIndex == opp_index and
                    log.toArea == AreaType.DISCARD and
                    log.cardId is not None):
                data = card_table.get(log.cardId)
                if data and data.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
                    reward += 0.05

            # Disruption: opponent's hand shuffled to deck (Iono / Judge style); capped at one bonus per observation
            if (not opp_disrupted and
                    log.playerIndex == opp_index and
                    log.fromArea == AreaType.HAND and
                    log.toArea == AreaType.DECK):
                opp_disrupted = True
                reward += 0.03

        # Bench/active damage: +0.01 per 10 HP lost by an opponent Pokémon
        elif (log.type == LogType.HP_CHANGE and
              log.playerIndex == opp_index and
              log.value is not None and log.value < 0):
            reward += 0.001 * (-log.value)

        # Paralysis landing on opponent
        elif (log.type == LogType.PARALYZED and
              log.playerIndex == opp_index and
              log.isRecover is False):
            reward += 0.03

        # Sleep landing on opponent
        elif (log.type == LogType.ASLEEP and
              log.playerIndex == opp_index and
              log.isRecover is False):
            reward += 0.03

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
                        reward -= 0.02

    return reward


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

# Single Training Sample
class LearnSample:
    def __init__(self, value: float, policy: list[float], sv_enc: SparseVector, sv_dec:SparseVector):
        self.value = value # Encoder output
        self.policy = policy # Decoder output
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec
   
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

    obs = search_state.observation
    state = obs.current
    if state.result >= 0:
        # Battle finished
        if state.result == 2:
            node.value = 0
        elif state.result == your_index:
            node.value = 1
        else:
            node.value = -1
        node.backprop(node.value)
        sample = None
    else:
        # For MAIN (maxCount=1): non-terminal actions first, then ATTACK, then END (always included)
        if obs.select.context == SelectContext.MAIN and obs.select.maxCount == 1:
            _terminal = (OptionType.ATTACK, OptionType.END)
            non_terminal = [i for i, o in enumerate(obs.select.option) if o.type not in _terminal]
            terminal    = [i for i, o in enumerate(obs.select.option) if o.type in _terminal]
            budget = 64 - len(terminal)
            actions = [[i] for i in non_terminal[:budget]] + [[i] for i in terminal]
        else:
            actions = []
            indices = list(range(obs.select.maxCount))
            for _ in range(64):
                actions.append(indices.copy())
                for i in range(len(indices)):
                    index = len(indices) - i - 1
                    if indices[index] < len(obs.select.option) - i - 1:
                        indices[index] += 1
                        for j in range(index+1, len(indices)):
                            indices[j] = indices[j - 1] + 1
                        break
                else:
                    break

        sv_enc = get_encoder_input(obs, your_deck)
        sv_dec = get_decoder_input(obs, actions)
        value, policy = eval_nn(sv_enc, sv_dec, model)
        v = value
        if state.yourIndex != your_index:
            v = -v
        v = max(-1.0, min(1.0, v + shaped_reward(obs, your_index)))
        node.value = v
        node.backprop(v)
        
        # --- Domain-Biased Priors (only when it is our player's turn) ---
        if state.yourIndex == your_index:
            your_ps = state.players[your_index]
            for i, action in enumerate(actions):
                if not action:
                    continue

                opt = obs.select.option[action[0]]

                # 1. Prioritize playing key Supporters / cards on turn 1
                if opt.type == OptionType.PLAY:
                    card = your_ps.hand[opt.index]
                    if card.id == 1220 and state.turn <= 2: # TR Proton on turn 1
                        policy[i] *= 1.5

                # 2. Prioritize selecting TR Proton from deck after Transceiver on turn 1
                elif (opt.type == OptionType.CARD
                      and obs.select.context == SelectContext.TO_HAND
                      and state.turn <= 2):
                    picked = get_card(obs, opt.area, opt.index, opt.playerIndex)
                    if picked and picked.id == 1220:
                        policy[i] *= 2.0

                # 3. Prioritize using Spidops Ability if it has no energy
                elif opt.type == OptionType.ABILITY:
                    pokemon_card = get_card(obs, opt.area, opt.index, your_index)
                    if pokemon_card and pokemon_card.id == 401: # Spidops
                        if not pokemon_card.energyCards:
                            policy[i] *= 1.6
        # --- End of Domain-Biased Priors ---

        prob_sum = 0.0
        for i in range(len(policy)):
            p = math.exp(policy[i] * 10.0)
            node.children.append(Child(actions[i], p))
            prob_sum += p
        for c in node.children:
            c.prob /= prob_sum
        sample = LearnSample(value, policy, sv_enc, sv_dec)

    return (node, sample)

# We will perform exploration using MCTS and select actions. At the same time, we will also generate training data.
def mcts_agent(obs_dict: dict, your_deck: list[int], model: MyModel) -> tuple[list[int], LearnSample]:
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    active = state.players[1 - your_index].active
    search_state = search_begin(
        obs,
        your_deck=random.sample(your_deck, state.players[your_index].deckCount), # Randomly select from deck.
        your_prize=random.sample(your_deck, len(state.players[your_index].prize)), # Randomly select from deck.
        opponent_deck=[1072] * state.players[1 - your_index].deckCount, # Fill with Snorlax (There is no deep meaning).
        opponent_prize=[1] * len(state.players[1 - your_index].prize), # Fill with Basic Energy (There is no deep meaning)
        opponent_hand=[1] * state.players[1 - your_index].handCount, # Fill with Basic Energy.
        opponent_active=[1072] if len(active) > 0 and active[0] == None else []) # Fill with Snorlax.
    root, sample = create_node(None, search_state, your_index, your_deck, model) # Create root node.

    # Search
    for _ in range(SEARCH_COUNT):
        current = root
        while True:
            value = -1e9
            c = 0.4 * math.sqrt(current.visit)
            for child in current.children:
                visit = 0
                if child.node == None:
                    v = current.total / current.visit
                else:
                    v = child.node.total / child.node.visit
                    visit = child.node.visit
                if current.state.observation.current.yourIndex != your_index:
                    v = -v
                v += c * child.prob / (1 + visit)
                if value < v:
                    value = v
                    next = child
            
            if next.node == None:
                search_state = search_step(current.state.searchId, next.select)
                next.node, _ = create_node(current, search_state, your_index, your_deck, model)
                break
            else:
                current = next.node
                if current.state.observation.current.result >= 0:
                    current.backprop(current.value)
                    break

    # Select the most visited node.
    max_child = None
    max_visit = -1
    min_value = 10
    for child in root.children:
        if child.node != None:
            if max_visit < child.node.visit:
                max_child = child
                max_visit = child.node.visit
            v = child.node.total / child.node.visit
            if min_value > v:
                min_value = v

    # Generate training data
    sample.value = root.total / root.visit
    for i in range(len(root.children)):
        child = root.children[i]
        v = sample.value
        if child.node == None:
            v = min_value - v - 0.03
        else:
            v = child.node.total / child.node.visit - v
        sample.policy[i] = max(-1.0, min(1.0, v))

    search_end()
    return (max_child.select, sample)


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

if __name__ == "__main__":
    # Juno's M2 List
    #sample_deck = [721,721,722,722,722,722,723,723,723,723,1092,1121,1121,1145,1145,1163,1163,1219,1219,1219,1219,1227,1227,1227,1227,1262,1262,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]

    dragapult = [119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 112, 112, 305, 66, 235, 140, 1071, 1227, 1227, 1227, 1227,1182, 1182, 1182, 1198, 1198, 1240, 1086, 1086, 1086, 1086, 1152, 1152, 1152, 1152, 1121, 1121, 1121, 1121, 1120, 1120, 1120, 1120, 1097, 1097, 1097, 1213, 1080, 1260, 1260, 5, 5, 5, 5, 2, 2, 2, 7, 7]
    grimmsnarl = [646, 646, 646, 646, 647, 647, 647, 648, 648, 648, 112, 112, 112, 112, 860, 860, 860, 104, 104, 104, 235, 235, 689, 1227, 1227, 1227, 1227, 1182, 1182, 1182, 1219, 1219, 1219, 1152, 1152, 1152, 1152, 1086, 1086, 1086, 1097, 1097, 1079, 1122, 1092, 1213, 1174, 1259, 1259, 1259, 1259, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    lucario = [677, 677, 677, 678, 678, 678, 676, 676, 676, 673, 673, 674, 674, 675, 675, 1071, 1227, 1227, 1227, 1227, 1213, 1213, 1182, 1182, 1211, 1219, 1229, 1142, 1142, 1142, 1142, 1152, 1152, 1152, 1152, 1121, 1121, 1121, 1121, 1141, 1141, 1141, 1141, 1080, 1174, 1174, 1174, 1252, 1252, 6, 6, 6, 6, 6, 6,6, 6, 6, 6, 6]
    mega_lopunny = [305, 305, 305, 65, 66, 66, 66, 848, 848, 848, 849, 849, 849, 109, 791, 174, 869, 1229, 1229, 1229, 1229, 1182, 1182, 1182, 1182, 1225, 1225, 1225, 1227, 1227, 1227, 1121, 1121, 1121, 1121, 1152, 1152, 1152, 1152, 1122, 1122, 1122, 1122, 1086, 1086, 1086, 1174, 1174, 1174, 1264, 1264, 1264, 11, 11, 11, 11, 16, 16, 16, 13]
    slop_box = [756, 756, 756, 756, 1071, 1071, 1071, 1071, 272, 272, 272, 272, 184, 184, 184, 108, 108, 140, 140, 791, 209, 979,1198, 1198, 1198, 1198, 1182, 1182, 1182, 1188, 1188, 1205, 1121, 1121, 1121, 1121, 1102, 1102, 1102, 1102, 1146, 1146, 1146, 1088, 1172, 1172, 1250, 1250, 1250, 1250, 5, 5, 5, 5, 3,3, 6, 6, 19, 2]

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
    WARMUP_EPOCHS             = 1  if FAST_TEST else 5
    WARMUP_SELF_PLAY_GAMES    = 3  if FAST_TEST else 50
    MAIN_EPOCHS               = 2  if FAST_TEST else 20
    EVAL_EVERY                = 1  if FAST_TEST else 5
    MAIN_SELF_PLAY_M2_GAMES   = 3  if FAST_TEST else 50
    MAIN_SELF_PLAY_OPP_GAMES  = 2  if FAST_TEST else 20
    MAIN_CROSS_PLAY_GAMES     = 2  if FAST_TEST else 10
    EVAL_GAMES_PER_MATCHUP    = 2  if FAST_TEST else 10

    class AgentState:
        def __init__(self, name: str, deck: list[int]):
            self.name = name
            self.deck = deck
            self.model = MyModel(128, 2, 256, 3, 1).to(device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
            self.replay: deque = deque(maxlen=REPLAY_BUFFER_MAXLEN)

        def save_checkpoint(self):
            folder = os.path.join("checkpoints", self.name)
            os.makedirs(folder, exist_ok=True)
            ts = time.strftime("%Y-%m-%d_%H-%M")
            path = os.path.join(folder, f"model_{ts}.pth")
            torch.save(self.model.state_dict(), path)
            print(f"[{self.name}] Saved: {path}")

        def load_latest_checkpoint(self):
            folder = os.path.join("checkpoints", self.name)
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

    def _backup_and_store(agent, result: int, player_idx: int, samples: list):
        value = 1.0 if player_idx == result else (0.0 if result == 2 else -1.0)
        for sample in reversed(samples):
            label = (value + sample.value) * 0.5
            value = value * LAMBDA + sample.value * (1.0 - LAMBDA)
            sample.value = label
            agent.replay.append(sample)

    def run_self_play(agent, n_games: int):
        agent.model.eval()
        with torch.inference_mode():
            for _ in progress(n_games, f"[{agent.name}] self-play"):
                obs, _ = battle_start(agent.deck, agent.deck)
                per_player: list[list[LearnSample]] = [[], []]
                while obs["current"]["result"] < 0:
                    yi = obs["current"]["yourIndex"]
                    selected, sample = mcts_agent(obs, agent.deck, agent.model)
                    per_player[yi].append(sample)
                    obs = battle_select(selected)
                battle_finish()
                result = obs["current"]["result"]
                for pi in range(2):
                    _backup_and_store(agent, result, pi, per_player[pi])

    def run_cross_play(m2, opp, n_games: int):
        m2.model.eval()
        opp.model.eval()
        with torch.inference_mode():
            for i in progress(n_games, f"[m2 vs {opp.name}]"):
                if i % 2 == 0:
                    obs, _ = battle_start(m2.deck, opp.deck)
                    by_player = [m2, opp]
                else:
                    obs, _ = battle_start(opp.deck, m2.deck)
                    by_player = [opp, m2]
                per_player: list[list[LearnSample]] = [[], []]
                while obs["current"]["result"] < 0:
                    yi = obs["current"]["yourIndex"]
                    selected, sample = mcts_agent(obs, by_player[yi].deck, by_player[yi].model)
                    per_player[yi].append(sample)
                    obs = battle_select(selected)
                battle_finish()
                result = obs["current"]["result"]
                for pi in range(2):
                    _backup_and_store(by_player[pi], result, pi, per_player[pi])

    def train_agent(agent):
        sample_list = list(agent.replay)
        if len(sample_list) < BATCH_SIZE:
            print(f"[{agent.name}] Too few samples ({len(sample_list)}), skipping.")
            return
        print(f"[{agent.name}] Training ({len(sample_list)} samples)...")
        agent.model.train()
        random.shuffle(sample_list)
        batch_count = len(sample_list) // BATCH_SIZE
        for i in range(batch_count):
            input_enc = LearnInput()
            input_dec = LearnInput()
            mask: list[float] = []
            label_enc: list[float] = []
            label_dec: list[float] = []
            for sample in sample_list[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]:
                input_enc.add(sample.sv_enc)
                input_dec.add(sample.sv_dec)
                label_enc.append(sample.value)
                label_dec.extend(sample.policy)
                mask.extend([1.0] * len(sample.policy))
                pad = 64 - len(sample.policy)
                mask.extend([0.0] * pad)
                label_dec.extend([0.0] * pad)
                for _ in range(pad):
                    input_dec.offset.append(len(input_dec.index))

            mask_t = torch.tensor(mask, dtype=torch.float32, device=device).view(BATCH_SIZE, -1)
            lbl_enc = torch.tensor(label_enc, dtype=torch.float32, device=device).view(BATCH_SIZE, -1)
            lbl_dec = torch.tensor(label_dec, dtype=torch.float32, device=device).view(BATCH_SIZE, -1)

            agent.optimizer.zero_grad()
            out_enc, out_dec = agent.model(
                torch.tensor(input_enc.index, dtype=torch.int32, device=device),
                torch.tensor(input_enc.value, dtype=torch.float32, device=device),
                torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
                torch.tensor(input_dec.index, dtype=torch.int32, device=device),
                torch.tensor(input_dec.value, dtype=torch.float32, device=device),
                torch.tensor(input_dec.offset, dtype=torch.int32, device=device))

            loss_enc = loss_fn_enc(out_enc, lbl_enc)
            loss_dec = (loss_fn_dec(out_dec, lbl_dec) * mask_t).sum() / mask_t.sum().clamp(min=1)
            (loss_enc + loss_dec).backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
            agent.optimizer.step()
        print(f"[{agent.name}] Done ({batch_count} batches).")

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
    all_agents = [
        AgentState("m2",           my_deck),
        AgentState("dragapult",    dragapult),
        AgentState("grimmsnarl",   grimmsnarl),
        AgentState("lucario",      lucario),
        AgentState("mega_lopunny", mega_lopunny),
        AgentState("slop_box",     slop_box),
    ]
    m2_agent        = all_agents[0]
    opponent_agents = all_agents[1:]

    for agent in all_agents:
        agent.load_latest_checkpoint()

    total_games = 0
    run_start = time.time()

    # === WARM-UP PHASE ===
    print(f"=== Warm-up Phase ({WARMUP_EPOCHS} epochs) ===")
    for epoch in range(WARMUP_EPOCHS):
        print(f"--- Warm-up Epoch {epoch + 1}/{WARMUP_EPOCHS} ---")
        for agent in all_agents:
            run_self_play(agent, WARMUP_SELF_PLAY_GAMES)
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
        run_self_play(m2_agent, MAIN_SELF_PLAY_M2_GAMES)
        total_games += MAIN_SELF_PLAY_M2_GAMES
        for opp in opponent_agents:
            run_self_play(opp, MAIN_SELF_PLAY_OPP_GAMES)
            total_games += MAIN_SELF_PLAY_OPP_GAMES

        # Cross-play: M2 vs each opponent (both sides collect samples)
        for opp in opponent_agents:
            run_cross_play(m2_agent, opp, MAIN_CROSS_PLAY_GAMES)
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
