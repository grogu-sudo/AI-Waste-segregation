import argparse
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import serial
from colorama import Fore, Style, init
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

init(autoreset=True)

CLASSES = ["glass", "metal", "organic", "plastic"]
COMMAND_MAP = {"WET": "W", "DRY": "D", "METAL": "M"}
LABEL_COLOR = {
    "WET": Fore.BLUE,
    "DRY": Fore.YELLOW,
    "METAL": Fore.CYAN,
    "WAITING": Fore.WHITE,
}


@dataclass
class SensorPacket:
    moisture: int
    ir: int
    fill_pct: int
    metal_sensor: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sensor + Camera fusion waste classifier"
    )
    parser.add_argument("--port", default="/dev/cu.usbserial-A5069RR4")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--model", default="models/image_model.keras")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=960)
    parser.add_argument("--camera-height", type=int, default=540)
    parser.add_argument("--sensor-moisture-threshold", type=int, default=980)
    parser.add_argument("--ir-detected-value", type=int, choices=[0, 1], default=0)
    parser.add_argument("--fill-alert-pct", type=int, default=85)
    parser.add_argument("--cooldown-sec", type=float, default=2.5)
    parser.add_argument("--sensor-timeout-sec", type=float, default=2.0)
    return parser.parse_args()


def get_center_roi(frame: np.ndarray, scale: float = 0.78) -> np.ndarray:
    h, w = frame.shape[:2]
    size = max(120, int(min(h, w) * scale))
    half = size // 2
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    return frame[y1:y2, x1:x2]


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    x = preprocess_input(resized)
    return np.expand_dims(x, axis=0)


def read_sensor_packet(ser: serial.Serial) -> SensorPacket | None:
    raw = ser.readline().decode("utf-8", errors="ignore").strip()
    if not raw or "READY" in raw or "TEST" in raw:
        return None
    parts = raw.split(",")
    if len(parts) != 4:
        return None
    try:
        moisture = int(parts[0])
        ir = int(parts[1])
        fill_pct = int(parts[2])
        metal_sensor = int(parts[3])
    except ValueError:
        return None
    return SensorPacket(moisture, ir, fill_pct, metal_sensor)


def object_detected(
    pkt: SensorPacket, moisture_threshold: int, ir_detected_value: int
) -> bool:
    # Manual-run mode: treat every cycle as "object present" and classify directly.
    return True


def image_probs_to_waste_probs(image_probs: np.ndarray) -> dict[str, float]:
    glass, metal, organic, plastic = [float(v) for v in image_probs]
    return {
        "WET": organic,
        "METAL": metal,
        "DRY": glass + plastic,
    }


def sensor_waste_probs(pkt: SensorPacket, moisture_threshold: int) -> dict[str, float]:
    if pkt.metal_sensor == 1:
        return {"WET": 0.01, "DRY": 0.01, "METAL": 0.98}
    if pkt.moisture < moisture_threshold:
        return {"WET": 0.92, "DRY": 0.07, "METAL": 0.01}
    if pkt.ir == 0:
        return {"WET": 0.10, "DRY": 0.85, "METAL": 0.05}
    return {"WET": 0.05, "DRY": 0.94, "METAL": 0.01}


def fuse_scores(
    sensor_scores: dict[str, float], image_scores: dict[str, float]
) -> tuple[str, float, dict[str, float]]:
    # Sensor remains strong for METAL safety; camera drives DRY/WET nuance.
    fused = {
        "WET": 0.35 * sensor_scores["WET"] + 0.65 * image_scores["WET"],
        "DRY": 0.35 * sensor_scores["DRY"] + 0.65 * image_scores["DRY"],
        "METAL": 0.65 * sensor_scores["METAL"] + 0.35 * image_scores["METAL"],
    }
    label = max(fused, key=fused.get)
    confidence = fused[label] * 100.0
    return label, confidence, fused


def infer_from_single_frame(model, frame: np.ndarray) -> np.ndarray:
    roi = get_center_roi(frame)
    if roi.size == 0:
        raise RuntimeError("Invalid ROI for classification")
    batch = preprocess_frame(roi)
    probs = model.predict(batch, verbose=0)[0]
    return probs


def draw_bar(value: int, width: int = 24) -> str:
    value = max(0, min(100, value))
    filled = int((value / 100.0) * width)
    bar = "█" * filled + "░" * (width - filled)
    color = Fore.GREEN if value < 60 else (Fore.YELLOW if value < 85 else Fore.RED)
    return color + f"[{bar}]" + Style.RESET_ALL


def print_box(
    pkt: SensorPacket,
    status: str,
    label: str,
    confidence: float,
    cmd: str,
    cam_top_label: str,
    cam_top_conf: float,
) -> None:
    os.system("cls" if os.name == "nt" else "clear")
    label_color = LABEL_COLOR.get(label, Fore.WHITE)
    print(Fore.CYAN + "╔════════════════════════════════════════════════════════════╗")
    print(Fore.CYAN + "║           AI WASTE SEGREGATOR — FUSION ENGINE             ║")
    print(Fore.CYAN + "╚════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Moisture : {Fore.WHITE}{pkt.moisture:>4}")
    print(f"  IR       : {Fore.WHITE}{pkt.ir:>4}")
    print(f"  Metal    : {Fore.WHITE}{pkt.metal_sensor:>4}")
    print(f"  Fill     : {draw_bar(pkt.fill_pct)} {pkt.fill_pct:>3}%")
    print()
    print("  ┌─ FINAL RESULT ───────────────────────────────────────────┐")
    print(f"  │  Status      : {Fore.WHITE}{status:<44}{Style.RESET_ALL}│")
    print(f"  │  Category    : {label_color}{label:<44}{Style.RESET_ALL}│")
    print(
        f"  │  Confidence  : {Fore.GREEN}{confidence:>6.1f}%{'':<37}{Style.RESET_ALL}│"
    )
    print(
        f"  │  Cam Top     : {Fore.MAGENTA}{cam_top_label:<12}{cam_top_conf:>6.1f}%{'':<24}{Style.RESET_ALL}│"
    )
    print(f"  │  Command     : {Fore.CYAN}{cmd:<44}{Style.RESET_ALL}│")
    print("  └──────────────────────────────────────────────────────────┘")
    print()
    print(Fore.WHITE + "  Press Ctrl+C to stop")


def draw_overlay(
    frame: np.ndarray,
    label: str,
    confidence: float,
    cmd: str,
    status: str,
    cam_top_label: str,
    cam_top_conf: float,
    fill_pct: int,
) -> None:
    h, _w = frame.shape[:2]
    panel_h = 140
    cv2.rectangle(frame, (0, 0), (760, panel_h), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Final: {label}  ({confidence:.1f}%)   Cmd: {cmd}",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Cam Top: {cam_top_label} ({cam_top_conf:.1f}%)   Fill: {fill_pct}%",
        (14, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 230, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Status: {status}",
        (14, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "Exit keys: Q / X / ESC",
        (14, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        1,
    )
    cv2.putText(
        frame,
        "Q/X/ESC to exit",
        (14, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
    )


def main() -> None:
    args = parse_args()
    print(f"Connecting Arduino on {args.port} @ {args.baud}...")
    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    time.sleep(2.0)
    print("Loading camera + image model...")
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cv2.namedWindow("Fusion Camera Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fusion Camera Preview", args.camera_width, args.camera_height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")
    model = load_model(args.model)

    print("Running fusion engine. Press Ctrl+C to stop.")
    last_sent = 0.0
    object_active = False
    last_label = "WAITING"
    last_conf = 0.0
    last_cmd = "-"
    last_status = "Waiting for object..."
    last_cam_label = "-"
    last_cam_conf = 0.0
    last_fill_pct = 0
    latest_frame = None
    last_sensor_time = time.time()

    try:
        while True:
            # Keep camera preview live every loop to avoid freeze.
            ret_preview, frame_preview = cap.read()
            if ret_preview:
                latest_frame = frame_preview.copy()
                h, w = frame_preview.shape[:2]
                size = max(120, int(min(h, w) * 0.78))
                half = size // 2
                cx, cy = w // 2, h // 2
                x1, y1 = max(0, cx - half), max(0, cy - half)
                x2, y2 = min(w, cx + half), min(h, cy + half)
                cv2.rectangle(frame_preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_preview,
                    f"{last_label} {last_conf:.1f}%",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                draw_overlay(
                    frame_preview,
                    label=last_label,
                    confidence=last_conf,
                    cmd=last_cmd,
                    status=last_status,
                    cam_top_label=last_cam_label,
                    cam_top_conf=last_cam_conf,
                    fill_pct=last_fill_pct,
                )
                cv2.imshow("Fusion Camera Preview", frame_preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), ord("x"), ord("X"), 27):
                    break

            pkt = None
            for _ in range(8):
                candidate = read_sensor_packet(ser)
                if candidate is None:
                    break
                pkt = candidate

            if pkt is None:
                if time.time() - last_sensor_time > args.sensor_timeout_sec:
                    last_status = "Waiting for sensor data..."
                continue

            last_sensor_time = time.time()
            last_fill_pct = pkt.fill_pct

            now = time.time()

            if object_active and (now - last_sent) < args.cooldown_sec:
                continue

            if pkt.fill_pct >= args.fill_alert_pct:
                ser.write(b"B")
                last_sent = now
                object_active = True
                last_label = "WAITING"
                last_conf = 0.0
                last_cmd = "B"
                last_status = f"Bin near full ({pkt.fill_pct}%)"
                last_cam_label = "-"
                last_cam_conf = 0.0
                print_box(
                    pkt,
                    last_status,
                    last_label,
                    last_conf,
                    last_cmd,
                    last_cam_label,
                    last_cam_conf,
                )
                continue

            if latest_frame is None:
                continue
            img_probs = infer_from_single_frame(model, latest_frame)
            img_scores = image_probs_to_waste_probs(img_probs)
            sens_scores = sensor_waste_probs(pkt, args.sensor_moisture_threshold)
            last_status = "Decision sent to Arduino"
            label, conf, _fused = fuse_scores(sens_scores, img_scores)

            # Hard safety rule:
            # classify as METAL only when inductive/metal sensor confirms it.
            if pkt.metal_sensor == 1:
                label = "METAL"
                conf = 100.0
            elif label == "METAL":
                # If camera/fusion suggests metal but sensor does not, pick best non-metal class.
                label = "WET" if _fused["WET"] >= _fused["DRY"] else "DRY"
                conf = _fused[label] * 100.0

            cmd = COMMAND_MAP[label]
            ser.write(cmd.encode("utf-8"))

            top_idx = int(np.argmax(img_probs))
            last_label = label
            last_conf = conf
            last_cmd = cmd
            last_cam_label = CLASSES[top_idx]
            last_cam_conf = img_probs[top_idx] * 100.0
            print_box(
                pkt,
                last_status,
                last_label,
                last_conf,
                last_cmd,
                last_cam_label,
                last_cam_conf,
            )
            object_active = True
            last_sent = now

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        ser.close()


if __name__ == "__main__":
    main()
