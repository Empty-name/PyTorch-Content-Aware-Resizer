import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from PIL import Image
import numpy as np

def load_image(path):
    """Load an image as a float tensor in [0, 1], shape (H, W, 3)."""
    img = Image.open(path).convert('RGB')
    return torch.from_numpy(np.array(img)).float() / 255.0

def to_grayscale(img):
    """Convert (H, W, 3) RGB tensor to (H, W) grayscale."""
    return 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]

def show(img, title=""):
    """Display a tensor image. Works for (H,W) grayscale or (H,W,3) RGB."""
    plt.figure(figsize=(10, 6))
    if img.dim() == 2:
        plt.imshow(img.numpy(), cmap='gray')
    else:
        plt.imshow(img.numpy().clip(0, 1))
    plt.title(title)
    plt.axis('off')
    plt.show()

def compute_energy(gray):
    """
    Compute the energy map of a grayscale image.

    Args:
        gray: (H, W) tensor, grayscale image in [0, 1]
    Returns:
        (H, W) tensor, energy map (non-negative)
    """
    Sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    Sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)

    gray_4d = gray.unsqueeze(0).unsqueeze(0)

    Gx = F.conv2d(gray_4d, Sx)
    Gy = F.conv2d(gray_4d, Sy)

    energy = torch.abs(Gx) + torch.abs(Gy)

    energy = energy.squeeze()

    energy = F.pad(energy.unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze(0)
    return energy

def find_seam(energy):
    """
    Find the minimum-energy vertical seam.

    Args:
        energy: (H, W) tensor
    Returns:
        seam: (H,) integer tensor of column indices
    """
    H, W = energy.shape
    M = torch.zeros_like(energy)
    M[0, :] = energy[0, :]
    for i in range(1, H):
        prev_row = F.pad(M[i - 1], (1, 1), value=float('inf'))
        M[i] = energy[i] + torch.min(prev_row[:-2], torch.min(prev_row[1:-1], prev_row[2:]))
    j = torch.argmin(M[-1])
    seam = torch.empty(H, dtype=torch.int)
    seam[H - 1] = j
    for i in range(H - 2, -1, -1):
        left = max(0, j - 1)
        right = min(W, j + 2)
        j = torch.argmin(M[i, left:right]) + left
        seam[i] = j
    return seam

def remove_seam(img, seam):
    """
    Remove a vertical seam from an image.

    Args:
        img: (H, W) or (H, W, 3) tensor
        seam: (H,) integer tensor of column indices to remove
    Returns:
        (H, W-1) or (H, W-1, 3) tensor
    """
    H, W = img.shape[:2]
    return img[torch.arange(W) != seam.unsqueeze(1)].reshape((H, W - 1) + img.shape[2:])

def seam_carve(img, n_seams, show_progress=True):
    """
    Remove n_seams vertical seams from img.

    Args:
        img: (H, W, 3) RGB tensor
        n_seams: number of seams to remove
        show_progress: print progress every 10 seams
    Returns:
        (H, W - n_seams, 3) tensor
    """
    for i in range(1, n_seams + 1):
        if show_progress and i % 10 == 0:
            print(f"{i} Seams removed", end="\r", flush=True)
        gray_img = to_grayscale(img)
        energy = compute_energy(gray_img)
        seam = find_seam(energy)
        img = remove_seam(img, seam)
    return img

def seam_insert(img, n_seams, show_progress=True):
    """
    Insert n_seams vertical seams into img (make it wider).

    Args:
        img: (H, W, 3) RGB tensor
        n_seams: number of seams to insert
        show_progress: print progress
    Returns:
        (H, W + n_seams, 3) tensor
    """
    H, W = img.shape[:2]
    orig_img = img.clone()
    if W == 1:
        return orig_img.expand(H, n_seams + 1, 3).clone()
    if n_seams == 0:
        return img.clone()

    orig_seams = []
    ind_map = torch.arange(W).expand(H, W)
    for i in range(1, n_seams + 1):
        if show_progress and i % 10 == 0:
            print(f"{i} Seams removed", end="\r", flush=True)
        gray_img = to_grayscale(img)
        energy = compute_energy(gray_img)
        seam = find_seam(energy)
        orig_seams.append(ind_map[torch.arange(H), seam])
        img = remove_seam(img, seam)
        ind_map = remove_seam(ind_map, seam)

    if show_progress:
        print()

    all_seams = torch.stack(orig_seams, dim=1).sort(descending=True).values
    new_img = []

    for i in range(H):
        if show_progress and i % 50 == 0:
            print(f"{i} Rows changed", end="\r", flush=True)
        row = orig_img[i]
        for j in all_seams[i]:
            if j > 0 and j < W - 1:
                new_pix = (row[j - 1] + row[j + 1]) / 2.0
            elif j == 0:
                new_pix = (row[0] + row[1]) / 2.0
            else:
                new_pix = (row[W - 2] + row[W - 1]) / 2.0
            row = torch.cat([row[:j + 1], new_pix.unsqueeze(0), row[j + 1:]], dim=0)
        new_img.append(row)
    if show_progress:
        print()
    return torch.stack(new_img)