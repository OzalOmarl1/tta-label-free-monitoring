# Checkpoints

The weights of the five source models (`resnet18_cifar10_seed42.pth` … `resnet18_cifar10_seed46.pth`, ~86 MB each) are not included in the repository. Download them from the `v1.0` release on the GitHub **Releases** page and place them in this folder.

Verification: the SHA256 values of the downloaded files must match `SHA256SUMS.txt` in the release and `Final/results/logs/checkpoint_hashes_final.json`.

```powershell
Get-FileHash .\resnet18_cifar10_seed42.pth -Algorithm SHA256
```
