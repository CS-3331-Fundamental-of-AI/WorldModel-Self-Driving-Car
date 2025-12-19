import torch

# ----------------------------
# Upsample Helper
# ----------------------------
def up2(x):
    return x.repeat_interleave(2, -1).repeat_interleave(2, -2)

def move_j1_to_device(batch_j1, device):
    """
    Move JEPA-1 batch data to the target device (CPU / GPU).

    Parameters
    ----------
    batch_j1 : tuple
        A tuple where each element corresponds to one JEPA-1 field.
        Each field is typically:
        - a list of torch.Tensors (batch dimension not stacked yet), or
        - a list of non-tensor data (numpy arrays, ints, metadata, etc.)

    Returns
    -------
    tuple
        Same structure as batch_j1, but with tensor lists moved to `device`.
        No stacking is done here.
    """

    new_batch = []

    for item in batch_j1:
        # Case 1: this field is a list/tuple of torch.Tensors
        # Example: [Tensor, Tensor, Tensor, ...]
        if isinstance(item, (list, tuple)) and isinstance(item[0], torch.Tensor):
            # Move each tensor in the list to the target device
            new_batch.append([x.to(device) for x in item])

        # Case 2: non-tensor data (numpy arrays, ints, metadata, etc.)
        # Keep it unchanged
        else:
            new_batch.append(item)

    # Return the same structure as input
    return tuple(new_batch)

