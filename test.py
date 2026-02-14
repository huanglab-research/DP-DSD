import requests

url = "https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_nodes4_gpu16_bs8_multicrop_epoch300_dino_aug_window14_lv/continued_from_epoch0200_dense_norm_true/checkpoint_best.pth?sp=r&st=2023-08-28T01:36:35Z&se=3023-08-28T09:36:35Z&sv=2022-11-02&sr=c&sig=coos9vSl4Xk6S6KvqZffkVCUb7Ug%2FFR9cfyc3xacMJI%3D"

response = requests.get(url)
print(response.status_code)
print(response.text if response.status_code != 200 else "Download success")
