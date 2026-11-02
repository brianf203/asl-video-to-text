#!/bin/bash
systemctl --user start tracker-miner-fs-3.service evolution-source-registry.service \
  evolution-calendar-factory.service evolution-addressbook-factory.service 2>&1
setsid -f python3 /usr/share/nvpmodel_indicator/nvpmodel_indicator.py >/dev/null 2>&1
setsid -f gnome-software --gapplication-service >/dev/null 2>&1
setsid -f update-notifier >/dev/null 2>&1
setsid -f /usr/libexec/evolution-data-server/evolution-alarm-notify >/dev/null 2>&1
setsid -f /usr/libexec/goa-daemon >/dev/null 2>&1
sleep 5; ps -u "$USER" -o comm | grep -c -E "tracker-miner|evolution|nvpmodel|gnome-software|update-notif|goa-daemon" | sed 's/^/running: /'
