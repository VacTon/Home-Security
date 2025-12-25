import cv2
import os
import time
import sys

# Add parent dir to path to import Camera
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from camera import Camera

def add_new_user():
    print("=== NEW USER REGISTRATION ===")
    name = input("Enter Name of New User: ").strip()
    if not name:
        print("Invalid name.")
        return

    save_dir = f"faces/{name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print(f"Warning: User '{name}' already exists. Appending photos.")

    print("Initializing Camera...")
    # Use standard VGA for capture
    cam = Camera(width=640, height=480)
    try:
        cam.start()
        time.sleep(2.0) # Warmup
        
        print(f"Prepare to capture 150 photos for {name}!")
        print("Please rotate your head slightly, smile, and change angles.")
        print("Press ENTER to start...")
        input()
        
        count = 0
        limit = 150
        
        while count < limit:
            frame = cam.get_frame()
            if frame is None:
                continue
            
            # Show preview
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Capturing: {count}/{limit}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Registration", display_frame)
            cv2.waitKey(1)
            
            # Save
            timestamp = int(time.time() * 1000)
            filename = f"{save_dir}/{name}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            count += 1
            print(f"Saved {count}/{limit}", end='\r')
            
            # Small delay to prevent identical frames
            time.sleep(0.05)
            
        print("\nCapture Complete!")
        print(f"Saved {count} photos to {save_dir}")
        print("Don't forget to run: python tools/process_database.py")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    add_new_user()
