#!/usr/bin/env pybricks-micropython
"""EV3 truck: read the laptop's Bluetooth Classic HID gamepad."""

import struct

import uselect
from pybricks import ev3brick as brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port, Stop
from pybricks.tools import StopWatch, wait


# Motor ports: A steers; B and C both drive the truck.
steering_motor = Motor(Port.A)
drive_motor_b = Motor(Port.B)
drive_motor_c = Motor(Port.C)

# The steering linkage is physically centered before starting this program.
STEER_LEFT_DEG = 45
STEER_RIGHT_DEG = -45
STEER_SPEED = 120

# The HID bridge changes this spare button every 250 ms as a heartbeat.
# No gamepad event for this long means stop, while keeping the event file open.
CONTROLLER_TIMEOUT_MS = 1500
# input_event.value is a signed 32-bit integer; joystick forward is negative.
EVENT_FORMAT = "llHHi"
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)


def stop_drive():
    # Stop each side independently so a temporary error on one port does not
    # prevent the other drivetrain motor from being stopped.
    for motor in (drive_motor_b, drive_motor_c):
        try:
            motor.dc(0)
        except OSError as error:
            print("Antrieb konnte nicht gestoppt werden:", error)


def center_steering():
    # The truck is physically centered when this program starts. When steering
    # returns to zero, coast instead of continuously holding the motor there.
    try:
        steering_motor.stop(Stop.COAST)
    except OSError as error:
        print("Lenkung konnte nicht freigegeben werden:", error)


def find_gamepad_event():
    """Find our HID gamepad by name; event2 is this EV3's fallback."""
    try:
        devices = open("/proc/bus/input/devices", "r")
        contents = devices.read()
        devices.close()
        for block in contents.split("\n\n"):
            if "EV3 Truck Gamepad" not in block:
                continue
            for line in block.split("\n"):
                if line.startswith("H: Handlers="):
                    for item in line.split():
                        if item.startswith("event"):
                            return "/dev/input/" + item
    except OSError:
        pass
    return "/dev/input/event2"


def signed_axis(value):
    """Normalize the standard signed HID axis range to -1..1."""
    value = max(-127, min(127, value))
    return value / 127.0


def handle_axis(code, value, previous_steering):
    if code == 3:  # ABS_RX: right stick X -> steering motor angle
        steering_value = signed_axis(value)
        if abs(steering_value) < 0.04:
            steering_value = 0
        if steering_value >= 0:
            target = int(STEER_LEFT_DEG * steering_value)
        else:
            target = int(STEER_RIGHT_DEG * abs(steering_value))
        # Ignore the HID bridge's one-count heartbeat dither. It must not make
        # the steering motor chatter around a held angle.
        if abs(target - previous_steering) >= 2:
            steering_motor.run_target(
                STEER_SPEED, target, then=Stop.COAST, wait=False
            )
            previous_steering = target
        elif target == 0 and previous_steering != 0:
            center_steering()
            previous_steering = 0

    elif code == 4:  # ABS_RY: right stick Y -> both drive motors
        # Positive signed HID Y is forward for this truck's motor mounting.
        # Keep the full -100..100 range so reverse remains available too.
        drive = max(-100, min(100, int(signed_axis(value) * 100)))
        drive_motor_b.dc(drive)
        drive_motor_c.dc(drive)

    return previous_steering


def run():
    brick.display.clear()
    brick.display.text("LKW: startet")
    steering_motor.reset_angle(0)
    center_steering()
    stop_drive()
    stopwatch = StopWatch()
    last_event_ms = stopwatch.time()
    previous_steering = 0

    print("Warte auf EV3 Truck Gamepad (Bluetooth Classic HID)")
    while True:
        event_path = find_gamepad_event()
        event_file = None
        try:
            event_file = open(event_path, "rb")
            poller = uselect.poll()
            poller.register(event_file, uselect.POLLIN)
            print("Gamepad-Eingang:", event_path)
            brick.display.clear()
            brick.display.text("LKW: Programm laeuft")
            brick.display.text("Bluetooth: verbunden")
            brick.display.text(event_path)

            while True:
                ready = poller.poll(100)
                if ready:
                    try:
                        event = event_file.read(EVENT_SIZE)
                    except OSError as error:
                        # Bluetooth input errors are not motor errors. Stop
                        # safely, close this event node, and reconnect below.
                        raise OSError("Gamepad-Eingabe verloren: " + str(error))
                    if not event:
                        raise OSError("Gamepad getrennt")
                    if len(event) != EVENT_SIZE:
                        continue
                    _sec, _usec, ev_type, code, value = struct.unpack(
                        EVENT_FORMAT, event
                    )
                    # EV_KEY includes the bridge heartbeat even when both
                    # stick axes stay still; any input event refreshes safety.
                    if ev_type == 1 or ev_type == 3:
                        last_event_ms = stopwatch.time()
                    if ev_type == 3:
                        previous_steering = handle_axis(
                            code, value, previous_steering
                        )

                if stopwatch.time() - last_event_ms > CONTROLLER_TIMEOUT_MS:
                    stop_drive()
                    if previous_steering != 0:
                        center_steering()
                        previous_steering = 0

        except OSError as error:
            print("Gamepad nicht verfügbar:", error)
            stop_drive()
            if previous_steering != 0:
                center_steering()
                previous_steering = 0
            wait(500)
        finally:
            if event_file is not None:
                try:
                    event_file.close()
                except Exception:
                    pass


run()
