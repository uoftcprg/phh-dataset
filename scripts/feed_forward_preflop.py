#!/usr/bin/env python3
# pylint: disable=line-too-long

"""
Feed-forward network to predict a player's pre-flop action (fold, check/call, raise).
Just an evaluation, not feature dense.

Reads pre-flop rows from the decisions/ table written by phh_to_parquet.py. Each row becomes 12 numbers:
    cards | known flag, high rank, low rank, suited, pair (all 0 when cards are hidden)
    table | seat (0 = small blind), n_players, n_active, n_raises
    chips | pot, to_call, and stack

Basic usage:
    ./scripts/feed_forward_preflop.py
"""

##### other TODOs ####################
# TODO: not attached to any one part of the code, just improvements to make:
# Infer hidden cards based on current representation distributions and what action the player took for better training quality.
#     most significantly, need to unroll the unknown's fold rates back into the weaker hands
#     (Kristian)
#
# Use an alternative hand encoding that better represents how strong the hands are sensitive to how people play them.
#     lets the model learn card strengths based on how people play rather than the mathematical significance
#     (Surja)
#
# Use all of the data, stream the rows instead of loading it all into memory.
#     should make this optional, so we can test with a smaller sample to get the other features implemented first
#     (Cole)
#
# Include player IDs into the decisions/ table and model learning
#     think this would be useful as a PoC that it can adapt to new players/data with non-traditional behavior
#     (Kirsten)
#
# More hidden layers & epochs
#     (Surja)
######################################

# std libs
import sys
import random
from pathlib import Path
from argparse import ArgumentParser

# external libs
import torch
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

##### constants ####################
ACTION_NAMES = ["fold", "call", "raise"]
COLUMNS = ["hand_id", "seat", "n_players", "n_active", "big_blind", "pot", "to_call", "stack", "n_raises", "hole1", "hole2", "action"]

# pre-flop only, and skip the bad rows: some sites record unknown stacks as inf, and negative blind entries make the pot negative
# TODO: fix or remove these bad rows... TODO Complete

## ROW_FILTERS = (pc.field("street") == 0) & pc.is_finite(pc.field("stack")) & (pc.field("pot") >= 0) - Original script
ROW_FILTERS = ( (pc.field("street")==0) & pc.is_finite(pc.field("big_blind")) & (pc.field("big_blind")>0) & pc.is_finite(pc.field("pot")) & (pc.field("pot")>=0) & pc.is_finite(pc.field("to_call")) & (pc.field("to_call")>=0) & pc.is_finite(pc.field("stack")) & (pc.field("stack")>=0))

# Extrapolated the filters that were already in place to the other conditions of the game. This should skip over the bad rows. - CRC

##### funcs ####################
def get_args():
    """ Argparser retrieval """

    parser = ArgumentParser(description=__doc__)

    parser.add_argument("--data_path", type = Path, help = "path to the decisions parquet table", default = "parquet/decisions")
    #parser.add_argument("--max_rows", type = int, help = "pre-flop rows to load", default = 10000000) # TODO: change to optional for final model (Cole)
    parser.add_argument("--max_rows", type = int, help = "pre-flop rows to load", default = None) # Changed default to None - CRC
    parser.add_argument("--epochs", type = int, help = "dataset iterations", default = 3)
    parser.add_argument("--threads", type = int, help = "torch threads", default = 2) # 2 seems to be safe on the VMs

    return parser.parse_args()


def load_preflop(data_path, max_rows):
    """ Reads pre-flop rows from randomly ordered parts """
    # TODO: fairly sure there's a better way to do this leveraging pyarrow, to research...
    #     (Kirsten)

    tables = []
    n_rows = 0

    files = sorted(data_path.rglob("*.parquet"))

    # shuffling the original dataset organization so they aren't grouped by stake
    # should remove the fixed seed if we don't want to change other configurations like Adam/ReLU
    random.Random(0).shuffle(files)

    for path in files:
        tables.append(pq.read_table(path, columns = COLUMNS, filters = ROW_FILTERS))
        n_rows += len(tables[-1]) # num rows in current table
        if max_rows is not None and n_rows >= max_rows: # Updated to account for the default change - CRC
            break
    table = pa.concat_tables(tables).slice(0, max_rows) # trim trailing rows beyond the max_rows
    if max_rows is not None:
        table = table.slice(0, max_rows) # Updated to Trim excess rows only when a limit was specificed (account for None)

    return { # parquet -> numpy -> Tensor based dict all in one, didn't see a direct parquet -> Tensor util
        column: torch.from_numpy(table[column].to_numpy()) for column in COLUMNS
    }

def stream_preflop(data_path, max_rows=None, batch_size=65536)
    """ Reads pre-flop data in batches instead of all at once """
    n_rows = 0 
    files = sorted(data_path.rglob("*.parquet"))

    # shuffling the original dataset organization so they aren't grouped by stake
    # should remove the fixed seed if we don't want to change other configurations like Adam/ReLU
    random.Random(0).shuffle(files)

    for path in files:
        # Open one Parquet file at a time 
        parquet_file = pq.ParquetFile(path)
        # Read up to 65,536 rows at a time.
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=COLUMNS+["street"]
        ):
            # Convert the batch to a table so we can filter it. 
            table = pa.Table.from_batches([batch])
            table = table.filter(ROW_FILTERS)
            # Skip batches that contain no valid pre-flop rows
            if table.num_rows ==0:
                continue
            # Limit the total number of rows if requested.
            if max_rows is not None:
                remaining = max_rows-n_rows

                if remaining <= 0:
                    return 

                table = table.slice(0, remaining)
            # Convert the current batch to PyTorch tensors
            columns = {
                column: torch.from_numpy(table[column].to-numpy())
                for column in COLUMNS
            }
            # Count only the rows we're actually yielding. 
            n_rows += table.num_rows
            # Send the batch to the training code. 
            yield columns 
            # Stop once the requested row limit is reached.
            if max_rows is not None and n_rows >= max_rows:
                return
def make_features(columns):
    """ Raw decision columns -> (rows, 12) float tensor """

    # in-house ref for card encodings:
    #   rank * 4 + suit
    #   - ranks 2 -> A as 0 -> 12 and suits c d h s as 0 -> 3.
    #       2c = 0  * 4 + 0 = 0
    #       2s = 0  * 4 + 3 = 3
    #       8d = 6  * 4 + 1 = 25
    #       As = 12 * 4 + 3 = 51

    hole1 = columns["hole1"]
    hole2 = columns["hole2"]
    known = hole1 >= 0

    # flooring gives us ranks; e.g. aces // 4 = 12
    rank1 = hole1 // 4
    rank2 = hole2 // 4

    big_blind = columns["big_blind"]

    # NOTE: organizing with dim = 1, rotates the decisions rows to append horizontally in the vector space
    return torch.stack([
        # card features; * known seeds 0s when the cards are hidden
        torch.maximum(rank1, rank2) / 12 * known, # always seed high card first
        torch.minimum(rank1, rank2) / 12 * known,
        ((hole1 % 4) == (hole2 % 4)) * known, # suited flag
        (rank1 == rank2) * known, # pair flag

        known, # flag
        # all divided by a scaling factor to get them between 0-1
        # should probably resolve this instead of going off the current dataset maximums
        columns["seat"] / 9,
        columns["n_players"] / 10,
        columns["n_active"] / 10,
        columns["n_raises"],

        # dividing to get the chip values as a relation to the big blind; idea from Cole
        # log1p is essentially "safe log" for the 0 case; using log to prevent the inevitable squish the same divising factor logic from before would provide
        #   since the vast majority of values in this dataset are less than the big blind, outliers would result in strong biases'
        # TODO: should test other normalization strategies to see if those make a significant difference
        #     (Kristian - deprioritized)
        torch.log1p(columns["pot"] / big_blind),
        torch.log1p(columns["to_call"] / big_blind),
        torch.log1p(columns["stack"] / big_blind),
    ], dim = 1)


def print_example_hands(model):
    """ Sanity check/hand samples: first to act in a 6 player game, nobody has raised, 100 big blind stacks """

    # put high card to the left, not resolving high card like in make_features
    sample_hands = {
        "AA": (51, 50),
        "AKs": (51, 47),
        "JTs": (39, 35),
        "72o": (20, 1),
        "??": (-1, -1)
    }

    n = len(sample_hands)

    # NOTE: .full((n,) ...) looks awkward but full expects a tuple or list and we only need 1D
    # https://docs.pytorch.org/docs/2.15/generated/torch.full.html
    columns = {
        "hole1": torch.tensor([hand[0] for hand in sample_hands.values()]),
        "hole2": torch.tensor([hand[1] for hand in sample_hands.values()]),
        "seat": torch.full((n,), 2), # first player after sm/bb
        "n_players": torch.full((n,), 6), # 6 players
        "n_active": torch.full((n,), 6),
        "n_raises": torch.zeros(n), # no raises
        "big_blind": torch.ones(n),
        "pot": torch.full((n,), 1.5), # just small + big blind
        "to_call": torch.ones(n),
        "stack": torch.full((n,), 100.0) # buy in default
    }

    with torch.no_grad():
        probablities = model(make_features(columns)).softmax(dim = 1)

    print("\nSelect hands sample")
    print(f"\t{"hand":<5}" + "".join(f"{action:>8}" for action in ACTION_NAMES))
    for hand, hand_probabilities in zip(sample_hands, probablities):
        print(f"\t{hand:<5}" + "".join(f"{prob:>8.2f}" for prob in hand_probabilities.tolist()))


def main():
    """ main driver - loads, trains, & reports """
    # TODO: convert most of this to a proper class structure, this isn't great
    #     (Kristian)

    args = get_args()
    torch.manual_seed(0) # fixed for now, could probably be an arg
    torch.set_num_threads(args.threads)

    columns = load_preflop(args.data_path, args.max_rows)
    features = make_features(columns)

    labels = columns["action"].long() # cross_entropy() requires long/byte, fails without
    known = columns["hole1"] >= 0 # -1 indicates unknown

    # splitting test/training subsets, just pulling about 1 in 10 for testing
    test_subset = columns["hand_id"] % 10 == 0
    train_in, train_target = features[~test_subset], labels[~test_subset]
    test_in, test_target = features[test_subset], labels[test_subset]

    print(f"Loaded {len(labels):,} pre-flop rows ({len(train_target):,} train, {len(test_target):,} test)")

    # Sequential for linear cxts - https://docs.pytorch.org/docs/2.15/generated/torch.nn.Sequential.html
    model = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], 64),
        torch.nn.ReLU(), # pretty much just using the most common features (which the baove source uses), ReLU happens to be suitable for this application
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, len(ACTION_NAMES)),
    )

    # Tuning handler, Adam appears to just be the most common optimizer, can tune if noted to be a limiting factor
    # default lr is already 1e-3 (from lib: `float | Tensor = 1e-3`), specified it because we may want to tune this
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    for _ in range(args.epochs):
        # further shuffling to mostly isolate the individual games into different batches
        # TODO: re-evaluate how many steps to split into once we ingest the full dataset, probably needs to be a function of N
        #     (Cole)
        for index in torch.randperm(len(train_target)).split(4000):
            loss = torch.nn.functional.cross_entropy(model(train_in[index]), train_target[index])
            optimizer.zero_grad()
            loss.backward() # backprop
            optimizer.step()

    # run a test to git the hit rate for accuracy
    with torch.no_grad(): # can more or less use no_grad() for all testing since we don't need to do backprop
        correct = model(test_in).argmax(dim = 1) == test_target

    print(f"\n{"test results":<14}{"rows":>12}{"accuracy":>10}")
    print(f"{"combined":<14}{len(correct):>12,}{correct.float().mean().item():>10.3f}")
    for name, group in [("cards known", known), ("cards unknown", ~known)]:
        test_group = group[test_subset]
        print(f"{name:<14}{test_group.sum().item():>12,}{correct[test_group].float().mean().item():>10.3f}")

    print_example_hands(model)


if __name__ == "__main__":
    sys.exit(main())
