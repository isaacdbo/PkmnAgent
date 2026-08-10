import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import torch
import RLTRM2 as R

N_GAMES, N_WORKERS, BASE_SEED = 200, 8, 7000000

def split(n, w):
    b, e = divmod(n, w)
    out, s = [], 0
    for i in range(w):
        sz = b + (1 if i < e else 0)
        out.append(list(range(s, s + sz))); s += sz
    return out

def main():
    deck = R.pd.read_excel("M2Deck.xlsx", header=None).iloc[:, 0].tolist()
    torch.manual_seed(1)
    model = R.MyModel(128, 2, 256, 3, 1); model.eval()
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    ctx = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx)
    try:
        warm = [pool.submit(R._self_play_worker, deck, state_dict, [], BASE_SEED) for _ in range(N_WORKERS)]
        for f in warm: f.result()
        t0 = time.perf_counter()
        slices = split(N_GAMES, N_WORKERS)
        futs = [pool.submit(R._self_play_worker, deck, state_dict, gi, BASE_SEED) for gi in slices]
        total = sum(f.result()[2] for f in futs)
        elapsed = time.perf_counter() - t0
    finally:
        pool.shutdown(wait=True)

    gph = total / elapsed * 3600
    print(f"workers={N_WORKERS} games={total} wall={elapsed:.2f}s games_per_hour={gph:.1f}")
    print(f"vs local N=8 baseline (2715.0/hr): {gph/2715.0:.2f}x")

if __name__ == "__main__":
    main()
