"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.models.base_model import BaseEncoder
from lavis.models.beats.BEATs import BEATs, BEATsConfig
import torch 
from lavis.common.utils import is_url
from lavis.common.dist_utils import download_cached_file
import os 


ckp_path =  "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D"

class BeatsEncoder(BaseEncoder):
    def __init__(self, checkpoint_path=ckp_path):
        super().__init__()
        
        # load the pre-trained checkpoints
        if is_url(checkpoint_path):
            cached_file = download_cached_file(
                checkpoint_path, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file)
        elif os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)

        cfg = BEATsConfig(checkpoint['cfg'])
        self.num_features = cfg.encoder_embed_dim
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    @classmethod
    def from_config(cls, cfg):
        checkpoint_path = cfg.get("checkpoint_path",ckp_path)
        return cls(checkpoint_path)

    def forward(self, x):
        with torch.no_grad():
            return self.model.extract_features(x.squeeze(1))[0]
    
