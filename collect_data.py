import csv
import serial
import time

PORT = "/dev/cu.usbserial-A5069RR4"
BAUD = 9600

ser = serial.Serial(PORT, BAUD, timeout=2)

time.sleep(2)

print("Connected to Arduino")

with open("data/sensor_dataset.csv", "a", newline="") as f:

    writer = csv.writer(f)

    while True:

        label = input("\nEnter label (WET/DRY/METAL or q): ")

        if label.lower() == "q":
            break

        print("Place object now...")

        raw = ser.readline().decode().strip()

        parts = raw.split(",")

        if len(parts) != 4:
            print("Invalid reading")
            continue

        moisture = int(parts[0])
        ir = int(parts[1])
        metal = int(parts[3])

        writer.writerow([moisture, ir, metal, label])

        print(
            f"Saved -> Moisture={moisture}, IR={ir}, Metal={metal}, Label={label}"
        )
