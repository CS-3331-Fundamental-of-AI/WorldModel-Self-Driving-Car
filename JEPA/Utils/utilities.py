# ----------------------------
# Upsample Helper
# ----------------------------
def up2(x):
    return x.repeat_interleave(2, -1).repeat_interleave(2, -2)

def maybe_to_device(x, device):
    if x is None:
        return None
    return x.to(device)