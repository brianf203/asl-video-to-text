#!/bin/bash
# User-session extras (no sudo). Reversed by start_user.sh.
systemctl --user stop tracker-miner-fs-3.service evolution-addressbook-factory.service \
  evolution-calendar-factory.service evolution-source-registry.service 2>&1
pkill -u "$USER" -f nvpmodel_indicator.py; pkill -u "$USER" -x gnome-software
pkill -u "$USER" -x update-notifier; pkill -u "$USER" -f evolution-alarm-notify; pkill -u "$USER" -x goa-daemon
sleep 3; ps -u "$USER" -o comm | grep -c -E "tracker-miner|evolution|nvpmodel|gnome-software|update-notif|goa-daemon" | sed 's/^/still running: /'
