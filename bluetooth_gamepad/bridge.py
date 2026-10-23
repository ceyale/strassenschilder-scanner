#!/usr/bin/env python3
"""Expose laptop traffic-sign commands as a Bluetooth Classic HID gamepad."""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

from bt_l2cap_v2 import L2CAPHIDServer


LOG = logging.getLogger("ev3_gamepad")
ROOT = Path(__file__).resolve().parent
COMMAND_PORT = int(os.environ.get("EV3_COMMAND_PORT", "8766"))
REPORT_INTERVAL = 0.05
COMMAND_TIMEOUT = 1.5
HEARTBEAT_INTERVAL = 0.25


def windows_host_address() -> str:
    """Find WSL2's NAT gateway, which is the Windows host from Ubuntu."""
    configured = os.environ.get("EV3_WINDOWS_HOST")
    if configured:
        return configured
    result = subprocess.run(
        ["ip", "route", "show", "default"],
        check=True,
        capture_output=True,
        text=True,
        timeout=3,
    )
    words = result.stdout.split()
    if len(words) < 3 or words[0] != "default" or words[1] != "via":
        raise RuntimeError("WSL Windows-host route not found; set EV3_WINDOWS_HOST")
    return words[2]


class GamepadState:
    """Standard HID Game Pad report; right stick axes map to EV3 ABS_RX/RY."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.body = bytearray(10)
        self.body[3] = 8  # neutral hat switch
        self.axis_x = 0
        self.axis_y = 0
        self.changed = True
        self.last_command = 0.0
        self.last_heartbeat = 0.0
        self.heartbeat = False

    def set_command(self, command: dict) -> None:
        try:
            power = int(command.get("drivePower", 0))
            angle = float(command.get("steeringAngle", 0))
        except (TypeError, ValueError):
            power, angle = 0, 0
        power = max(0, min(100, power))
        angle = max(-45.0, min(45.0, angle))
        # HID signed axis: -127 is full forward, +127 is full reverse.
        right_y = -round(power * 127 / 100)
        right_x = round(angle * 127 / 45)
        with self.lock:
            if self.axis_x != right_x or self.axis_y != right_y:
                self.axis_x = right_x
                self.axis_y = right_y
                self.changed = True
            self.last_command = time.monotonic()

    def neutral_if_stale(self) -> None:
        with self.lock:
            if time.monotonic() - self.last_command <= COMMAND_TIMEOUT:
                return
            if self.axis_x != 0 or self.axis_y != 0:
                self.axis_x = 0
                self.axis_y = 0
                self.changed = True

    def report_if_due(self) -> bytes | None:
        """Emit changed axes and a button-24 heartbeat for EV3's watchdog."""
        now = time.monotonic()
        with self.lock:
            if now - self.last_heartbeat >= HEARTBEAT_INTERVAL:
                self.last_heartbeat = now
                self.heartbeat = not self.heartbeat
                if self.heartbeat:
                    self.body[2] |= 0x80
                else:
                    self.body[2] &= 0x7F
                self.changed = True
            if not self.changed:
                return None
            self.changed = False
            # Some EV3 Linux kernels suppress unchanged EV_ABS events. Nudge
            # each axis by one raw count around its target so a program that
            # starts after Bluetooth pairing still receives the held command.
            nudge = 1 if self.heartbeat else -1
            x = max(-127, min(127, self.axis_x + nudge))
            y = max(-127, min(127, self.axis_y + nudge))
            self.body[8] = x & 0xFF
            self.body[9] = y & 0xFF
            # HIDP DATA | INPUT, report ID 1, followed by the 10-byte body.
            return b"\xA1\x01" + bytes(self.body)

    def mark_changed(self) -> None:
        with self.lock:
            self.changed = True


class CommandClient(threading.Thread):
    """Read the existing Windows backend's JSON/TCP command stream."""

    def __init__(self, state: GamepadState) -> None:
        super().__init__(name="windows-command-client", daemon=True)
        self.state = state
        self.stopping = threading.Event()

    def run(self) -> None:
        while not self.stopping.is_set():
            connection = None
            try:
                host = windows_host_address()
                LOG.info("Connecting to Windows backend %s:%d", host, COMMAND_PORT)
                connection = socket.create_connection((host, COMMAND_PORT), timeout=3)
                connection.settimeout(0.25)
                LOG.info("Windows backend connected")
                buffer = b""
                while not self.stopping.is_set():
                    try:
                        chunk = connection.recv(4096)
                    except socket.timeout:
                        self.state.neutral_if_stale()
                        continue
                    if not chunk:
                        raise ConnectionError("Windows backend closed the command stream")
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if line:
                            command = json.loads(line.decode("utf-8"))
                            if isinstance(command, dict):
                                self.state.set_command(command)
            except Exception as exc:
                self.state.set_command({"drivePower": 0, "steeringAngle": 0})
                self.state.last_command = 0.0
                LOG.warning("Backend unavailable; gamepad held neutral: %s", exc)
                self.stopping.wait(1.0)
            finally:
                if connection is not None:
                    try:
                        connection.close()
                    except OSError:
                        pass


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    state = GamepadState()
    gamepad = L2CAPHIDServer(
        sdp_record_path=str(ROOT / "sdp_record_gamepad.xml"),
        alias="EV3 Truck Gamepad",
    )
    gamepad.on_connect = state.mark_changed
    commands = CommandClient(state)
    commands.start()
    gamepad.start()
    LOG.info("Pair the EV3 with 'EV3 Truck Gamepad' over Bluetooth Classic")

    def report_loop() -> None:
        while True:
            report = state.report_if_due()
            if report is not None:
                gamepad.send_report(report)
            time.sleep(REPORT_INTERVAL)

    reports = threading.Thread(target=report_loop, name="hid-report-pump", daemon=True)
    reports.start()
    try:
        # Profile/agent callbacks require GLib to own the main thread.
        gamepad.serve()
    except KeyboardInterrupt:
        pass
    finally:
        commands.stopping.set()
        gamepad.stop()


if __name__ == "__main__":
    run()
