from config import PATCH_SIZE

def patchify(imgs, patch_size=PATCH_SIZE):
    """
    imgs: (B, C, H, W)
    returns tokens: (B, N, token_dim), ph, pw
    """
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0
    ph = H // patch_size
    pw = W // patch_size
    x = imgs.reshape(B, C, ph, patch_size, pw, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, ph * pw, C * patch_size * patch_size)
    return x, ph, pw

def unpatchify(tokens, ph, pw, patch_size=PATCH_SIZE):
    """
    tokens: (B, N, token_dim)
    returns imgs: (B, C, H, W)
    """
    B, N, token_dim = tokens.shape
    C = token_dim // (patch_size * patch_size)
    x = tokens.reshape(B, ph, pw, C, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, ph * patch_size, pw * patch_size)
    return x
