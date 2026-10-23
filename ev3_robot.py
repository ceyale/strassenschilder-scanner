#!/usr/bin/env pybricks-micropython
"""Connect an EV3 running Pybricks MicroPython to server.py over local Wi-Fi."""

import json
import socket
from pybricks.ev3devices import Motor
from pybricks.parameters import Port, Stop
from pybricks.tools import wait


# Set this to the laptop's Wi-Fi IPv4 address (the same laptop running server.py).
LAPTOP_HOST = "192.168.141.144"
LAPTOP_PORT = 8766

# The steering linkage is centered before this program starts. Change these
# values after a slow, wheels-off-the-floor calibration if the turn is too wide
# or the direction is reversed.
STEER_LEFT_DEG = 45
STEER_RIGHT_DEG = -45
STEER_SPEED = 120

# Set either sign to -1 if that drive motor is mounted in the opposite direction.
DRIVE_B_SIGN = 1
DRIVE_C_SIGN = 1

steering = Motor(Port.A)
drive_b = Motor(Port.B)
drive_c = Motor(Port.C)
steering.reset_angle(0)


def stop_drive():
    drive_b.dc(0)
    drive_c.dc(0)


def apply_command(command, previous_angle):
    power = command.get("drivePower", 0)
    if not isinstance(power, (int, float)):
        power = 0
    power = max(0, min(100, int(power)))
    drive_b.dc(power * DRIVE_B_SIGN)
    drive_c.dc(power * DRIVE_C_SIGN)

    angle = command.get("steeringAngle", 0)
    if not isinstance(angle, (int, float)):
        angle = 0
    if angle > 0:
        angle = STEER_LEFT_DEG * min(1, angle / 45)
    elif angle < 0:
        angle = STEER_RIGHT_DEG * min(1, abs(angle) / 45)
    angle = int(angle)
    if angle != previous_angle:
        steering.run_target(STEER_SPEED, angle, then=Stop.HOLD, wait=False)
    return angle


def run():
    while True:
        connection = None
        previous_angle = None
        try:
            print("Verbinde mit Laptop:", LAPTOP_HOST, LAPTOP_PORT)
            connection = socket.socket()
            connection.settimeout(2)
            connection.connect((LAPTOP_HOST, LAPTOP_PORT))
            print("Verbunden. Warte auf Fahrbefehle.")
            buffer = b""
            while True:
                try:
                    chunk = connection.recv(512)
                except OSError as error:
                    if "timed out" not in str(error).lower() and "timeout" not in str(error).lower():
                        raise
                    # Safety pause without closing the TCP connection. The next
                    # command can resume motion on this same connection.
                    print("Sicherheitspause: keine TCP-Daten; Verbindung bleibt offen.")
                    stop_drive()
                    steering.run_target(STEER_SPEED, 0, then=Stop.HOLD, wait=False)
                    previous_angle = None
                    continue
                if not chunk:
                    raise OSError("Laptop-Verbindung geschlossen")
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line:
                        continue
                    command = json.loads(line.decode("utf-8"))
                    previous_angle = apply_command(command, previous_angle)
                    reason = command.get("stopReason", "")
                    if reason:
                        print("Stopp:", reason)
                    else:
                        print("Fahrt:", command.get("drivePower", 0), "%")
        except Exception as error:
            print("Verbindung/Fahrbefehl fehlgeschlagen:", error)
            stop_drive()
            # On a broken connection, return the steering to its centered start.
            try:
                steering.run_target(STEER_SPEED, 0, then=Stop.HOLD, wait=False)
            except Exception:
                pass
            wait(1500)
        finally:
            if connection:
                try:
                    connection.close()
                except Exception:
                    pass


run()
