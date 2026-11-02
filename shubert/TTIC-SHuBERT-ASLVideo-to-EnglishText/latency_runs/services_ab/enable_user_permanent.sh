#!/bin/bash
for f in gnome-software-service nvpmodel_indicator org.gnome.Evolution-alarm-notify tracker-miner-fs-3 update-notifier; do
  rm -f ~/.config/autostart/$f.desktop; done
systemctl --user unmask tracker-miner-fs-3.service evolution-addressbook-factory.service \
  evolution-calendar-factory.service evolution-source-registry.service 2>&1
