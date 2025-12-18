import torch

# ----------------------------
# Upsample Helper
# ----------------------------
def up2(x):
    return x.repeat_interleave(2, -1).repeat_interleave(2, -2)

def move_j1_to_device(batch_j1, device):
    """
    batch_j1: tuple of lists, e.g., (bev_list, mask_emp_list, ...)
    Each element is a list of tensors for the batch.
    """
    new_batch = []
    for tensor_list in batch_j1:
        if isinstance(tensor_list[0], torch.Tensor):
            # stack along batch dimension and move to device
            new_batch.append(torch.stack(tensor_list, dim=0).to(device))
        else:
            # numpy arrays or numbers: keep as list
            new_batch.append(tensor_list)
    return tuple(new_batch)
