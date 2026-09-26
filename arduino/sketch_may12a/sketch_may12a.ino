#include <Servo.h>
#include <Stepper.h>

const int MOISTURE_PIN = A0;
const int IR_PIN = 5;
const int METAL_PIN = 6;
const int SERVO_PIN = 7;
const int BUZZER_PIN = 13;

// HC-SR04 ultrasonic
const int TRIG_PIN = 2;
const int ECHO_PIN = 3;

const int STEP_IN1 = 8;
const int STEP_IN3 = 9;
const int STEP_IN2 = 10;
const int STEP_IN4 = 11;

const int STEPS_PER_REV = 2048;
const int POS_WET = 0;
const int POS_DRY = STEPS_PER_REV / 3;
const int POS_METAL = (STEPS_PER_REV * 2) / 3;

// Tune using your real bin dimensions.
const float BIN_EMPTY_DISTANCE_CM = 28.0;
const float BIN_FULL_DISTANCE_CM = 7.0;

Servo lidServo;
Stepper sorter(STEPS_PER_REV, STEP_IN1, STEP_IN3, STEP_IN2, STEP_IN4);

int currentPosition = 0;
unsigned long lastSend = 0;
volatile bool stopRequested = false;

bool checkStopCommand() {
  if (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == 'X') {
      stopRequested = true;
      return true;
    }
  }
  return false;
}

bool waitWithStop(unsigned long ms) {
  unsigned long start = millis();
  while (millis() - start < ms) {
    if (checkStopCommand()) {
      return true;
    }
    delay(5);
  }
  return false;
}

bool stepToPositionWithStop(int targetPosition) {
  int stepsNeeded = targetPosition - currentPosition;
  int direction = (stepsNeeded >= 0) ? 1 : -1;
  int steps = abs(stepsNeeded);

  for (int i = 0; i < steps; i++) {
    if (checkStopCommand()) {
      return false;
    }
    sorter.step(direction);
  }
  currentPosition = targetPosition;
  return true;
}

float readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 25000);
  if (duration <= 0) {
    return -1.0;
  }
  return (duration * 0.0343) / 2.0;
}

int computeFillPercent() {
  float d = readDistanceCm();
  if (d < 0.0) {
    return 0;
  }

  if (d > BIN_EMPTY_DISTANCE_CM) d = BIN_EMPTY_DISTANCE_CM;
  if (d < BIN_FULL_DISTANCE_CM) d = BIN_FULL_DISTANCE_CM;

  float pct = (BIN_EMPTY_DISTANCE_CM - d) * 100.0 /
              (BIN_EMPTY_DISTANCE_CM - BIN_FULL_DISTANCE_CM);
  if (pct < 0.0) pct = 0.0;
  if (pct > 100.0) pct = 100.0;
  return (int)pct;
}

void setup() {
  Serial.begin(9600);

  pinMode(IR_PIN, INPUT);
  pinMode(METAL_PIN, INPUT_PULLUP);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  lidServo.attach(SERVO_PIN);
  lidServo.write(0);

  sorter.setSpeed(12);

  Serial.println("READY");
}

void loop() {
  if (millis() - lastSend >= 600) {
    int moisture = analogRead(MOISTURE_PIN);
    int irVal = digitalRead(IR_PIN);
    int metal = (digitalRead(METAL_PIN) == LOW) ? 1 : 0;
    int fillPct = computeFillPercent();

    Serial.print(moisture);
    Serial.print(",");
    Serial.print(irVal);
    Serial.print(",");
    Serial.print(fillPct);
    Serial.print(",");
    Serial.println(metal);

    lastSend = millis();
  }

  if (Serial.available() > 0) {
    char cmd = (char)Serial.read();
    switch (cmd) {
      case 'W':
        stopRequested = false;
        dumpWaste(POS_WET);
        break;
      case 'D':
        stopRequested = false;
        dumpWaste(POS_DRY);
        break;
      case 'M':
        stopRequested = false;
        dumpWaste(POS_METAL);
        break;
      case 'B':
        buzzAlert();
        break;
      case 'X':
        stopRequested = true;
        lidServo.write(0);
        break;
    }
  }
}

void dumpWaste(int targetPosition) {
  if (stopRequested) return;

  lidServo.write(0);
  if (waitWithStop(100)) return;

  if (!stepToPositionWithStop(targetPosition)) return;

  if (waitWithStop(100)) return;
  lidServo.write(120);
  if (waitWithStop(2000)) {
    lidServo.write(0);
    return;
  }
  lidServo.write(0);
  if (waitWithStop(400)) return;

  if (!stepToPositionWithStop(0)) return;
}

void buzzAlert() {
  for (int i = 0; i < 4; i++) {
    tone(BUZZER_PIN, 1200, 250);
    delay(350);
  }
}
