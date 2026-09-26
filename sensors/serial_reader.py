import serial

PORT = "/dev/cu.usbserial-A5069RR4"
BAUD = 9600

ser = serial.Serial(PORT, BAUD, timeout=2)


def read_sensor_data():

    raw = ser.readline().decode().strip()

    parts = raw.split(",")

    if len(parts) != 4:
        return None

    return {
        "moisture": int(parts[0]),
        "ir": int(parts[1]),
        "fill": int(parts[2]),
        "metal": int(parts[3])
    }
