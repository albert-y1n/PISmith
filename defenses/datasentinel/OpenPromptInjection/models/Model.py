import random
import os
import torch
import numpy as np


class Model:
    def __init__(self, config):
        self.provider = config["model_info"]["provider"]
        self.name = config["model_info"]["name"]
        self.seed = int(config["params"]["seed"])
        self.temperature = float(config["params"]["temperature"])
        self.gpus = config["params"]["gpus"]
        self._normalize_gpu_config()
        self.initialize_seed()
        if len(self.gpus) > 0:
            self.initialize_gpus()
        self.set_decoding_method(config)

    def print_model_info(self):
        print(f"{'-'*len(f'| Model name: {self.name}')}\n| Provider: {self.provider}\n| Model name: {self.name}\n| Do sample: {self.do_sample}\n{'-'*len(f'| Model name: {self.name}')}")

    def set_API_key(self):
        raise NotImplementedError("ERROR: Interface doesn't have the implementation for set_API_key")
    
    def query(self):
        raise NotImplementedError("ERROR: Interface doesn't have the implementation for query")
    
    def initialize_seed(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            # Respect actual visible device count, not raw config length.
            if torch.cuda.device_count() > 1:
                torch.cuda.manual_seed_all(self.seed)
    
    def initialize_gpus(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        # If caller already pinned visible devices, do not override it here.
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in self.gpus])

    def _normalize_gpu_config(self):
        """Normalize configured GPU list to current visible CUDA context."""
        # If visible devices are already set by launcher, follow that contract.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            visible_count = len([d for d in visible.split(",") if d.strip() != ""])
            if visible_count > 0:
                # Reindex to logical IDs under current CUDA_VISIBLE_DEVICES.
                self.gpus = [str(i) for i in range(visible_count)]
                return

        # Otherwise sanitize configured IDs by probing actual CUDA device count.
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            normalized = []
            for gpu in self.gpus:
                try:
                    gid = int(gpu)
                    if 0 <= gid < device_count:
                        normalized.append(str(gid))
                except (TypeError, ValueError):
                    continue
            self.gpus = normalized
        else:
            self.gpus = []
    
    def set_decoding_method(self, config):
        self.do_sample = True
        if "decoding_method" in config["params"]:
            self.decoding_method = config["params"]["decoding_method"]
            if self.decoding_method == 'greedy':
                self.do_sample = False