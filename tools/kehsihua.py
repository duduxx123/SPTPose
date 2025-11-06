import torch
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tokenpose_last_transformer_layer_density(
    model,
    bins: int = 50,
    binwidth: float = None,  # 如果不为 None，则按固定宽度分箱
    kde: bool = True
):
    """
    从 TokenPose-B 模型的最后一层 Transformer 中提取权重，
    并绘制注意力 qkv、proj 以及前馈 fc1、fc2 的密度直方图。

    参数:
      model: TokenPose-B 模型实例
      bins: 分箱数（当 binwidth=None 时生效）
      binwidth: 分箱宽度（如果不为 None，则覆盖 bins 设置）
      kde: 是否叠加 KDE 曲线
    """
    # 定位到最后一层 Transformer block
    transformer = model.keypoint_head.tokenpose.transformer_layer1
    last_block = transformer.layers[-1]

    # 提取权重
    attn = last_block[0].fn.fn
    ffn  = last_block[1].fn.fn
    weights = {
        "attn_qkv":   attn.to_qkv.weight.detach().cpu().numpy().flatten(),
        "attn_proj":  attn.to_out[0].weight.detach().cpu().numpy().flatten(),
        "ffn_fc1":    ffn.net[0].weight.detach().cpu().numpy().flatten(),
        "ffn_fc2":    ffn.net[3].weight.detach().cpu().numpy().flatten(),
    }

    # 绘图布局
    num = len(weights)
    cols = 2
    rows = (num + cols - 1) // cols
    plt.figure(figsize=(6 * cols, 4 * rows))

    # 绘制每个权重的 Density 直方图
    for idx, (name, w) in enumerate(weights.items()):
        ax = plt.subplot(rows, cols, idx + 1)

        hist_kwargs = dict(stat="density", kde=kde, ax=ax)
        if binwidth is not None:
            hist_kwargs["binwidth"] = binwidth
        else:
            hist_kwargs["bins"] = bins

        sns.histplot(w, **hist_kwargs)  # stat="density" 会归一化面积为 1 :contentReference[oaicite:0]{index=0}
        ax.set_title(f"{name} (Density)")
        ax.set_ylabel("Density")

    plt.tight_layout()
    plt.show()
