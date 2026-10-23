#!/bin/sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "Run this installer as root: sudo ./install-wsl.sh" >&2
    exit 1
fi

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y bluez python3-dbus python3-gi

install -d -m 0755 /opt/ev3-gamepad
install -m 0644 "$ROOT/bridge.py" /opt/ev3-gamepad/bridge.py
install -m 0644 "$ROOT/bt_l2cap_v2.py" /opt/ev3-gamepad/bt_l2cap_v2.py
install -m 0644 "$ROOT/BluezAgent.py" /opt/ev3-gamepad/BluezAgent.py
install -m 0644 "$ROOT/BluezProfile.py" /opt/ev3-gamepad/BluezProfile.py
install -m 0644 "$ROOT/hid_descriptor.py" /opt/ev3-gamepad/hid_descriptor.py
install -m 0644 "$ROOT/sdp_record_gamepad.xml" /opt/ev3-gamepad/sdp_record_gamepad.xml
install -D -m 0644 "$ROOT/bluetooth-override.conf" \
    /etc/systemd/system/bluetooth.service.d/override.conf
install -D -m 0644 "$ROOT/ev3-gamepad.service" \
    /etc/systemd/system/ev3-gamepad.service

systemctl daemon-reload
systemctl enable bluetooth.service ev3-gamepad.service
systemctl restart bluetooth.service
sleep 2
btmgmt power on
btmgmt connectable on
btmgmt bondable on
systemctl restart ev3-gamepad.service
systemctl --no-pager --full status ev3-gamepad.service
