#!/bin/bash
# Configures a 2GB zram swap optimized for a 4GB Raspberry Pi 4B

echo "Setting up zram..."
sudo modprobe zram
sudo zramctl --find --size 2G -a lz4
sudo mkswap /dev/zram0
sudo swapon /dev/zram0 -p 10

echo "zram configured successfully!"
sudo zramctl
