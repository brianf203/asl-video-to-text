#!/bin/bash
# Run with sudo. Temporary (systemctl stop, not disable): everything comes back on reboot or start_root.sh.
systemctl stop docker.socket docker.service containerd.service packagekit.service \
  snapd.socket snapd.service fwupd.service snap.cups.cupsd.service snap.cups.cups-browsed.service \
  ModemManager.service bluetooth.service
sync; echo 3 > /proc/sys/vm/drop_caches
systemctl is-active docker containerd packagekit snapd fwupd snap.cups.cupsd ModemManager bluetooth | paste -sd' '
