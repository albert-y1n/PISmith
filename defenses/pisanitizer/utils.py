import copy
import torch
import warnings


def top_k_mean(tensor_list, k=5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = torch.tensor(tensor_list)
        # t: [num_layers, seq_len]
        topk, _ = torch.topk(t, k=min(k, t.shape[0]), dim=0)
        return topk.float().mean(dim=0).tolist() 


def nice_print(text):
    print_text = copy.deepcopy(text)
    print_text = print_text.replace("\\", "\\\\").replace("\n", "\\n")
    print(print_text)


def process_attn(attentions, inputs):
    try:
        input_ids = inputs['input_ids']
    except:
        input_ids = inputs

    layer_max_attn = []
    layer_avg_attn = []
    layer_top5_attn = []
    assert attentions[0].shape[2] == 1, f"Invalid shape: {attentions[0].shape}"
    for layer_idx in range(len(attentions)):
        attn = attentions[layer_idx][0, :, :, :input_ids.shape[1]].mean(dim=1) # bsz, heads, seq_len, seq_len
        # attn = attentions[layer_idx][0, :, -1, :] # bsz, heads, seq_len, seq_len

        max_attn = attn.max(dim=0)[0] # max and average over attention heads
        avg_attn = attn.mean(dim=0)
        top5_attn = top_k_mean(attn, k=5)

        assert max_attn.shape[0] == input_ids.shape[1]
        assert avg_attn.shape[0] == input_ids.shape[1]
        assert len(top5_attn) == input_ids.shape[1]

        layer_max_attn.append(max_attn.detach().float().cpu().tolist())
        layer_avg_attn.append(avg_attn.detach().float().cpu().tolist())
        layer_top5_attn.append(top5_attn)
    return layer_max_attn, layer_avg_attn, layer_top5_attn


def remove_indices(tensor: torch.Tensor, idx_list: list) -> torch.Tensor:
    if tensor.dim() == 1:
        mask = torch.ones(tensor.size(0), dtype=torch.bool)
        mask[idx_list] = False
        return tensor[mask]
    else:
        raise ValueError(f"Invalid tensor dimension: {tensor.dim()}")
