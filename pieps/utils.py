from torch_geometric.data import Data


def shallow_copy(data):
    out =  Data(x=data.x.clone(), edge_index=data.edge_index, edge_attr=data.edge_attr, pos=data.pos, batch=data.batch)
    for key in ["active_clusters", "_changed_attr", "_changed_attr_indices","diff_idx", "diff_pos_idx", "pooling", "num_image_channels", "skipped", "pooled"]:
        if hasattr(data, key):
            setattr(out, key, getattr(data, key))
    for key in ["diff_idx", "diff_pos_idx"]:
        if hasattr(data, key):
            setattr(out, key, getattr(data, key).clone())
    return out
