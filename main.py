import time

from sensors.serial_reader import read_sensor_data
from ai.predict_sensor import predict_waste
from fusion.decision_engine import decide_command
from ui.dashboard import show_dashboard

from sensors.serial_reader import ser


while True:

    sensor_data = read_sensor_data()

    if sensor_data is None:
        continue

    moisture = sensor_data["moisture"]
    ir = sensor_data["ir"]
    metal = sensor_data["metal"]

    label, confidence = predict_waste(
        moisture,
        ir,
        metal
    )

    command = decide_command(label)

    ser.write(command.encode())

    show_dashboard(
        sensor_data,
        label,
        confidence,
        command
    )

    time.sleep(1)
