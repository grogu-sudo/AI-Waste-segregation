
def show_dashboard(sensor_data, label, confidence, command):

    print("\n============================")
    print(" SMART WASTE SEGREGATOR")
    print("============================")

    print(f"Moisture : {sensor_data['moisture']}")
    print(f"IR       : {sensor_data['ir']}")
    print(f"Metal    : {sensor_data['metal']}")

    print("----------------------------")

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.2f}%")
    print(f"Command    : {command}")
