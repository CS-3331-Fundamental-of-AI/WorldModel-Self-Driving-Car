import numpy as np
import random
from collections import deque
import time
import tools
import torch



def add_transition(replay, episode_id, transition, dataset_size):
    """
    Append transition to replay.
    Prune old episodes when dataset_size is exceeded.
    """
    tools.add_to_cache(replay, episode_id, transition)
    tools.erase_over_episodes(replay, dataset_size)
    
def extract_image(obs):
    """
    Extract raw image as a numpy array (H,W,C).
    Never uses boolean checks on arrays.
    """
    if isinstance(obs, dict):
        if "image" in obs:
            img = obs["image"]
        elif "pixel_values" in obs:
            img = obs["pixel_values"]
        else:
            raise KeyError("obs dict has no 'image' or 'pixel_values' key")
    else:
        img = obs

    # Safety checks (recommended)
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected image as np.ndarray, got {type(img)}")
    if img.dtype == object:
        raise TypeError("Image has dtype=object, replay cannot store this")

    return img

def build_transition(
    obs,
    action,
    reward,
    discount,
    is_first: bool,
    is_terminal: bool,
    is_last: bool,
):
    """
    Transition for JEPA-1 + RSSM (Dreamer-style).

    obs: dict-like (e.g. {"image": np.ndarray})
    action: [A] or None
    reward: scalar
    discount: scalar (0.0 if terminal else 1.0)
    is_first: True if episode reset
    is_terminal: True if env terminated
    is_last: True if this is final transition of episode
    """

    out = dict(obs)

    # --- action ---
    if action is not None:
        out["action"] = np.asarray(action, dtype=np.float32)

    # --- scalars ---
    out["reward"] = np.asarray(reward, dtype=np.float32)
    out["discount"] = np.asarray(discount, dtype=np.float32)

    # --- episode structure (CRITICAL) ---
    out["is_first"] = np.asarray(is_first, dtype=np.bool_)
    out["is_terminal"] = np.asarray(is_terminal, dtype=np.bool_)
    out["is_last"] = np.asarray(is_last, dtype=np.bool_)

    return out

def count_available_sequences(replay, valid_eps, T):
    """
    How many distinct length-T slices exist in replay across valid_eps.
    If this >= batch_size, you can form a full batch without waiting.
    """
    total = 0
    for ep_id in list(valid_eps):
        if ep_id not in replay:
            continue
        L = len(replay[ep_id].get("reward", []))
        total += max(0, L - T + 1)
    return total




class SequenceReplay:
    def __init__(self, capacity, seq_len):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        """
        transition: dict with keys
        image, action, reward, discount, is_terminal
        """
        self.buffer.append(transition)

    def can_sample(self, batch_size):
        return len(self.buffer) >= self.seq_len * batch_size

    def sample(self, batch_size):
        """
        Returns batch with shape [B, T, ...]
        """
        sequences = []

        for _ in range(batch_size):
            start = random.randint(
                0, len(self.buffer) - self.seq_len
            )
            seq = list(self.buffer)[start : start + self.seq_len]
            sequences.append(seq)

        batch = {}
        for key in sequences[0][0].keys():
            batch[key] = np.stack(
                [[step[key] for step in seq] for seq in sequences],
                axis=0,  # [B,T,...]
            )

        return batch
    

def sample_fixed_length_from_replay(replay, valid_eps, T):
    """Return dict of arrays [T,...] or None if not ready."""
    if not valid_eps:
        return None

    ep_id = random.choice(tuple(valid_eps))
    episode = replay.get(ep_id, None)
    if episode is None:
        return None

    # IMPORTANT: replay stores lists per key, so np.asarray(list) is fine.
    L = len(episode["reward"])
    if L < T:
        return None

    start = random.randint(0, L - T)

    sample = {}
    for k, v in episode.items():
        arr = np.asarray(v)
        sample[k] = arr[start:start + T]

    # enforce invariants
    sample["is_first"] = np.zeros((T, 1), np.float32)
    sample["is_first"][0, 0] = 1.0

    if "is_terminal" not in sample:
        sample["is_terminal"] = np.zeros((T, 1), np.float32)

    if "is_last" not in sample:
        sample["is_last"] = np.zeros((T, 1), np.float32)
        sample["is_last"][-1, 0] = 1.0

    if "discount" not in sample:
        sample["discount"] = np.ones((T, 1), np.float32)

    return sample


def fixed_length_generator(replay, valid_eps, T):
    """Never blocks: yields None until data is ready."""
    while True:
        yield sample_fixed_length_from_replay(replay, valid_eps, T)
        

def from_generator(
    generator,
    batch_size,
    name="Collecting samples",
    max_tries=5000,
    sleep_time=0.01,
):
    """
    Robust online batcher.
    Pulls fixed-length [T,...] samples into [T,B,...] Dreamer batches.
    Never blocks forever; yields only when a full batch is ready.
    """

    while True:
        batch = []
        tries = 0

        p_collect = tqdm(
            total=batch_size,
            desc=f"[GEN] {name}",
            leave=False,
        )

        # -----------------------------
        # Collect B samples safely
        # -----------------------------
        while len(batch) < batch_size and tries < max_tries:
            sample = next(generator)
            tries += 1

            if sample is None:
                time.sleep(sleep_time)
                p_collect.set_postfix(waiting="replay not ready")
                continue

            if not isinstance(sample, dict):
                continue

            batch.append(sample)
            p_collect.update(1)
            p_collect.set_postfix(collected=len(batch))

        p_collect.close()

        # -----------------------------
        # Not ready → wait and retry
        # -----------------------------
        if len(batch) < batch_size:
            print(
                f"⚠️ [GEN] Only collected {len(batch)}/{batch_size} samples "
                f"(replay not ready)"
            )
            time.sleep(0.1)
            continue

        # -----------------------------
        # Stack batch → [B,T,...]
        # -----------------------------
        first_val = batch[0]["discount"]   # stable Dreamer key
        T = len(first_val)

        all_keys = set().union(*(s.keys() for s in batch))
        data = {}

        for key in all_keys:
            values = []
            for i, s in enumerate(batch):
                v = np.asarray(s[key])
                if v.shape[0] != T:
                    raise RuntimeError(
                        f"[GEN] time mismatch key '{key}' "
                        f"sample {i}: expected {T}, got {v.shape[0]}"
                    )
                values.append(v)

            data[key] = np.stack(values, axis=0)  # [B,T,...]

        # -----------------------------
        # Ensure Dreamer scalar shapes
        # -----------------------------
        for k in ("reward", "discount", "is_first", "is_terminal", "is_last"):
            if k in data:
                if data[k].ndim == 2:
                    data[k] = data[k][..., None].astype(np.float32)
                else:
                    data[k] = data[k].astype(np.float32)

        # -----------------------------
        # Convert to time-major [T,B,...]
        # -----------------------------
        for k, v in data.items():
            if isinstance(v, np.ndarray) and v.ndim >= 2:
                data[k] = np.swapaxes(v, 0, 1)

        yield data
        
def preprocess(batch, device):
    # Move everything to torch
    batch = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}

    # ---- is_first: MUST be float [B,T,1] ----
    if "is_first" in batch:
        batch["is_first"] = batch["is_first"].float()

    # ---- is_terminal: keep semantic but convert ----
    if "is_terminal" in batch:
        batch["is_terminal"] = batch["is_terminal"].float()

        # cont = 1 - terminal  (Dreamer convention)
        batch["cont"] = 1.0 - batch["is_terminal"]

    return batch