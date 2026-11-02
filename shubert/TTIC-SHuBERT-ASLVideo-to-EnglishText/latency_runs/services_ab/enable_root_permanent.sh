#!/bin/bash
systemctl unmask packagekit.service
systemctl enable --now containerd.service docker.service packagekit.service snapd.service fwupd.service \
  snap.cups.cupsd.service snap.cups.cups-browsed.service ModemManager.service bluetooth.service
