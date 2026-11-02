#!/bin/bash
# Run with sudo. Permanent: disables at boot AND stops now. docker.socket / snapd.socket stay ENABLED, so
# `docker` or launching a snap (Firefox/Chromium) still starts its daemon on demand. Undo: enable_root_permanent.sh
systemctl disable --now docker.service containerd.service packagekit.service snapd.service fwupd.service \
  snap.cups.cupsd.service snap.cups.cups-browsed.service ModemManager.service bluetooth.service
systemctl mask packagekit.service    # otherwise D-Bus activation restarts it
systemctl is-enabled docker containerd packagekit snapd fwupd snap.cups.cupsd ModemManager bluetooth | paste -sd' '
