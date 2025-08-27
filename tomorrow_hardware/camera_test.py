#!/usr/bin/env python3
"""
CAMERA TEST - Super Simple Face Detection
Just plug in USB webcam and run!
"""

import time
import sys

print("📸 CAMERA TEST - Let's see if we can see!")
print("-" * 40)

# Check for camera libraries
HAS_CV2 = False
HAS_GPIO = False

try:
    import cv2
    HAS_CV2 = True
    print("✅ OpenCV ready")
except ImportError:
    print("⚠️  No OpenCV - install with:")
    print("    sudo apt install python3-opencv")
    sys.exit(1)

try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
    LED_PIN = 17
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)
    print("✅ GPIO ready - LED will show when face detected")
except:
    print("⚠️  No GPIO - running without LED feedback")

def simple_camera_test():
    """Just show camera feed"""
    print("\n1. BASIC CAMERA TEST")
    print("-" * 20)
    
    cap = cv2.VideoCapture(0)  # 0 = first camera
    
    if not cap.isOpened():
        print("❌ Can't open camera!")
        print("   Check: Is camera plugged in?")
        print("   Try: ls /dev/video*")
        return False
    
    print("✅ Camera opened!")
    print("   Press 'q' to quit")
    print("   Press 's' to save snapshot")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Can't read frame")
            break
        
        # Show frame
        cv2.imshow('Cypher Vision Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('snapshot.jpg', frame)
            print("📸 Saved snapshot.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def face_detection_test():
    """Detect faces and blink LED"""
    print("\n2. FACE DETECTION TEST")
    print("-" * 20)
    
    # Load face detector (comes with OpenCV)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    if face_cascade.empty():
        print("❌ Can't load face detector")
        return False
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Can't open camera")
        return False
    
    print("✅ Face detection ready!")
    print("   LED will light when face detected")
    print("   Press 'q' to quit")
    
    face_count = 0
    last_face_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw rectangles and light LED
        if len(faces) > 0:
            if HAS_GPIO:
                GPIO.output(LED_PIN, GPIO.HIGH)
            
            # Track first detection
            if time.time() - last_face_time > 2:
                face_count += 1
                print(f"👤 Face detected! (Total: {face_count})")
                last_face_time = time.time()
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "FACE", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            if HAS_GPIO:
                GPIO.output(LED_PIN, GPIO.LOW)
        
        # Show status on frame
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Cypher Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Stats: Detected faces {face_count} times")
    return True

def motion_detection_test():
    """Simple motion detection"""
    print("\n3. MOTION DETECTION TEST")
    print("-" * 20)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Can't open camera")
        return False
    
    print("✅ Motion detection ready!")
    print("   LED will blink on motion")
    
    # Read first frame
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    
    motion_events = 0
    
    while True:
        if not ret:
            break
        
        # Calculate difference
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        
        # Find contours (moving objects)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < 900:
                continue
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        if motion_detected:
            motion_events += 1
            cv2.putText(frame1, "MOTION!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if HAS_GPIO:
                GPIO.output(LED_PIN, GPIO.HIGH)
        else:
            if HAS_GPIO:
                GPIO.output(LED_PIN, GPIO.LOW)
        
        cv2.imshow('Cypher Motion Detection', frame1)
        
        # Update frames
        frame1 = frame2
        ret, frame2 = cap.read()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Stats: Detected {motion_events} motion events")
    return True

def main():
    print("\n" + "=" * 40)
    print("CYPHER VISION TESTS")
    print("=" * 40)
    
    # Check if camera exists
    import os
    if os.path.exists('/dev/video0'):
        print("✅ Camera device found at /dev/video0")
    else:
        print("⚠️  No /dev/video0 - camera might be at /dev/video1")
    
    tests = [
        ("Basic Camera", simple_camera_test),
        ("Face Detection", face_detection_test),
        ("Motion Detection", motion_detection_test)
    ]
    
    for name, test_func in tests:
        print(f"\nRunning: {name}")
        input("Press Enter to start...")
        
        try:
            result = test_func()
            if result:
                print(f"✅ {name} passed!")
            else:
                print(f"❌ {name} failed")
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
        
        print("\n" + "-" * 40)
    
    if HAS_GPIO:
        GPIO.cleanup()
    
    print("\n🎉 Vision tests complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        if HAS_GPIO:
            GPIO.cleanup()
        cv2.destroyAllWindows()