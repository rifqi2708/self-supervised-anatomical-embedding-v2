# A class for using SAM to find corresponding points
# Run in the project folder level
# %run miscellaneous/PMCC/matcher.py


"""PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.

Purpose: PMCC workflow runner or directly required PMCC helper, kept together following Rifqi's scope clarification.
Inputs: caller arguments and existing script settings.
Outputs: return values or existing export destinations.
Dependencies: numpy, os, shared.
Role: main runner.
Usage: the README in this workflow directory; existing CLI/settings remain applicable.
"""

# Direct-file entry points resolve imports from the repository root.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = 'miscellaneous.PMCC'
import numpy as np
from shared.utils import read_image
from shared.interfaces import init, get_embedding, get_sim_embed_loc
import os


class Match:
    def __init__(self, im1_file, im2_file, model) -> None:
        self.im1_file = im1_file
        self.im2_file = im2_file
        self.model = model
        self.im1, self.normed_im1, self.norm_info_1 = read_image(im1_file, is_MRI=False)
        self.im2, self.normed_im2, self.norm_info_2 = read_image(im2_file, is_MRI=False)

        self.emb1 = get_embedding(self.normed_im1, self.model)
        self.emb2 = get_embedding(self.normed_im2, self.model)

    def match(self, pt1, return_score=False):
        pt1_normed = np.array(pt1) * self.norm_info_1
        im2_shape = self.im2["shape"]
        pt2_normed, score = get_sim_embed_loc(
            self.emb1,
            self.emb2,
            pt1_normed,
            (im2_shape[3], im2_shape[1], im2_shape[2]),
            norm_info=self.norm_info_2,
            write_sim=False,
            use_sim_coarse=True,
        )
        pt2 = np.array(pt2_normed).astype(int)
        return (pt2, score) if return_score else pt2


if __name__ == "__main__":
    # Input files - two CTs
    im1_file = "data/raw_data/NIH_lymph_node/ABD_LYMPH_001.nii.gz"
    im2_file = "data/raw_data/NIH_lymph_node/ABD_LYMPH_002.nii.gz"

    # Model
    os.chdir(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )  # go to root dir of this project
    config_file = "configs/sam/sam_NIHLN.py"
    checkpoint_file = "checkpoints/SAM.pth"
    model = init(config_file, checkpoint_file)

    matcher = Match(im1_file, im2_file, model)

    pt1 = np.array([93, 139, 44])
    pt2 = matcher.match(pt1)

    print(pt2)
