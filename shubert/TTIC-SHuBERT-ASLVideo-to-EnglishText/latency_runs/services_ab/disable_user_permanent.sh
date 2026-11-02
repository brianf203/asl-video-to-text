#!/bin/bash
# Permanent (survives reboot) user-side half of the services stop. No sudo. Undo: enable_user_permanent.sh
mkdir -p ~/.config/autostart
for f in gnome-software-service nvpmodel_indicator org.gnome.Evolution-alarm-notify tracker-miner-fs-3 update-notifier; do
  printf '[Desktop Entry]\nType=Application\nName=%s\nHidden=true\n' "$f" > ~/.config/autostart/$f.desktop
done
systemctl --user mask tracker-miner-fs-3.service evolution-addressbook-factory.service \
  evolution-calendar-factory.service evolution-source-registry.service 2>&1
