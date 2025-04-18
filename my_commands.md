# Useful Commands

## Set GPU in terminal
This command sets the GPU to be used in the terminal.
```sh
export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
```

## Monitor GPU Usage
This command monitors GPU usage and updates every second (only the bottom processes part).
```sh
watch -n 1 "nvidia-smi | awk '/Processes:/ {flag=1; next} flag'"
```

```
pip install nvitop
nvitop
```

## List Open Files on NVIDIA Devices
This command lists all open files on NVIDIA devices.
```sh
lsof /dev/nvidia*
```