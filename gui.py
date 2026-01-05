import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import os
import sys
import signal

class SecuritySystemGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Home Security System Control Panel")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Process tracking
        self.main_process = None
        self.is_running = False
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🏠 AI Security System", 
                               font=('Arial', 20, 'bold'))
        title_label.grid(row=0, column=0, pady=10)
        
        # Control Buttons Frame
        control_frame = ttk.LabelFrame(main_frame, text="System Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Button grid layout
        btn_width = 20
        
        # Row 1: Main system controls
        self.start_btn = ttk.Button(control_frame, text="▶ Start System", 
                                   command=self.start_system, width=btn_width)
        self.start_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹ Stop System", 
                                  command=self.stop_system, width=btn_width, 
                                  state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Row 2: User management
        ttk.Button(control_frame, text="👤 Add New User (Auto-Process)", 
                  command=self.add_user, width=btn_width).grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(control_frame, text="🔄 Rebuild Database", 
                  command=self.process_database, width=btn_width).grid(row=1, column=1, padx=5, pady=5)
        
        # Row 3: Utilities
        ttk.Button(control_frame, text="📁 Open Faces Folder", 
                  command=self.open_faces_folder, width=btn_width).grid(row=2, column=0, padx=5, pady=5)
        
        ttk.Button(control_frame, text="📸 Open Strangers Folder", 
                  command=self.open_strangers_folder, width=btn_width).grid(row=2, column=1, padx=5, pady=5)
        
        # Row 4: Configuration
        ttk.Button(control_frame, text="⚙️ Edit Config", 
                  command=self.edit_config, width=btn_width).grid(row=3, column=0, padx=5, pady=5)
        
        ttk.Button(control_frame, text="📊 System Info", 
                  command=self.show_system_info, width=btn_width).grid(row=3, column=1, padx=5, pady=5)
        
        # Status Frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding="10")
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        
        # Status indicator
        self.status_label = ttk.Label(status_frame, text="● System Stopped", 
                                     font=('Arial', 12), foreground='red')
        self.status_label.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Log output
        self.log_text = scrolledtext.ScrolledText(status_frame, height=15, 
                                                  wrap=tk.WORD, font=('Courier', 9))
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Clear log button
        ttk.Button(status_frame, text="Clear Log", 
                  command=self.clear_log).grid(row=2, column=0, pady=5)
        
        # Initial log message
        self.log("Security System Control Panel initialized")
        self.log("Ready to start")
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def log(self, message):
        """Add message to log with timestamp"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_log(self):
        """Clear the log text"""
        self.log_text.delete(1.0, tk.END)
    
    def start_system(self):
        """Start the main security system"""
        if self.is_running:
            messagebox.showwarning("Already Running", "System is already running!")
            return
        
        self.log("Starting security system...")
        
        # Start main.py in subprocess
        try:
            self.main_process = subprocess.Popen(
                [sys.executable, "main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            self.is_running = True
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.status_label.config(text="● System Running", foreground='green')
            
            # Start thread to read output
            threading.Thread(target=self.read_process_output, daemon=True).start()
            
            self.log("System started successfully!")
            
        except Exception as e:
            self.log(f"ERROR: Failed to start system: {e}")
            messagebox.showerror("Error", f"Failed to start system:\n{e}")
    
    def read_process_output(self):
        """Read output from main process"""
        if self.main_process:
            for line in iter(self.main_process.stdout.readline, ''):
                if line:
                    self.log(line.strip())
                if self.main_process.poll() is not None:
                    break
            
            # Process ended
            self.is_running = False
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.status_label.config(text="● System Stopped", foreground='red')
            self.log("System stopped")
    
    def stop_system(self):
        """Stop the main security system"""
        if not self.is_running or not self.main_process:
            return
        
        self.log("Stopping security system...")
        
        try:
            # Send SIGTERM (graceful shutdown)
            if os.name == 'nt':  # Windows
                self.main_process.terminate()
            else:  # Linux/Mac
                self.main_process.send_signal(signal.SIGTERM)
            
            # Wait for process to end
            self.main_process.wait(timeout=5)
            
        except subprocess.TimeoutExpired:
            # Force kill if not responding
            self.log("Force stopping...")
            self.main_process.kill()
        except Exception as e:
            self.log(f"ERROR: {e}")
        
        self.is_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="● System Stopped", foreground='red')
        self.log("System stopped")
    
    def add_user(self):
        """Run add_user.py tool"""
        self.log("Launching Add User tool...")
        
        try:
            # Run in new terminal window
            if os.name == 'nt':  # Windows
                subprocess.Popen(['start', 'cmd', '/k', 'python', 'tools/add_user.py'], 
                               shell=True)
            else:  # Linux/Mac
                subprocess.Popen(['x-terminal-emulator', '-e', 'python3', 'tools/add_user.py'])
            
            self.log("Add User tool opened in new window")
            
        except Exception as e:
            self.log(f"ERROR: {e}")
            messagebox.showerror("Error", f"Failed to launch Add User:\n{e}")
    
    def process_database(self):
        """Run process_database.py"""
        self.log("Processing face database...")
        
        def run_process():
            try:
                result = subprocess.run(
                    [sys.executable, 'tools/process_database.py'],
                    capture_output=True,
                    text=True
                )
                
                self.log("--- Database Processing Output ---")
                self.log(result.stdout)
                if result.stderr:
                    self.log(f"ERRORS:\n{result.stderr}")
                self.log("--- Processing Complete ---")
                
                if result.returncode == 0:
                    messagebox.showinfo("Success", "Database processed successfully!")
                else:
                    messagebox.showerror("Error", "Database processing failed. Check log.")
                    
            except Exception as e:
                self.log(f"ERROR: {e}")
                messagebox.showerror("Error", f"Failed to process database:\n{e}")
        
        # Run in background thread
        threading.Thread(target=run_process, daemon=True).start()
    
    def open_faces_folder(self):
        """Open faces directory in file explorer"""
        try:
            if os.name == 'nt':  # Windows
                os.startfile('faces')
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', 'faces'])
            else:  # Linux
                subprocess.Popen(['xdg-open', 'faces'])
            
            self.log("Opened faces folder")
        except Exception as e:
            self.log(f"ERROR: {e}")
            messagebox.showerror("Error", f"Failed to open folder:\n{e}")
    
    def open_strangers_folder(self):
        """Open strangers directory in file explorer"""
        try:
            # Create folder if it doesn't exist
            if not os.path.exists('strangers'):
                os.makedirs('strangers')
            
            if os.name == 'nt':  # Windows
                os.startfile('strangers')
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', 'strangers'])
            else:  # Linux
                subprocess.Popen(['xdg-open', 'strangers'])
            
            self.log("Opened strangers folder")
        except Exception as e:
            self.log(f"ERROR: {e}")
            messagebox.showerror("Error", f"Failed to open folder:\n{e}")
    
    def edit_config(self):
        """Open config.yaml in default editor"""
        try:
            if os.name == 'nt':  # Windows
                os.startfile('config.yaml')
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', 'config.yaml'])
            else:  # Linux
                subprocess.Popen(['xdg-open', 'config.yaml'])
            
            self.log("Opened config.yaml")
            messagebox.showinfo("Config", "After editing config, restart the system for changes to take effect.")
        except Exception as e:
            self.log(f"ERROR: {e}")
            messagebox.showerror("Error", f"Failed to open config:\n{e}")
    
    def show_system_info(self):
        """Show system information"""
        info = []
        
        # Check if models exist
        if os.path.exists('models/w600k_r50.onnx'):
            info.append("✓ ArcFace model found")
        else:
            info.append("✗ ArcFace model missing")
        
        # Check if database exists (correct path)
        if os.path.exists('faces/encodings.pkl'):
            info.append("✓ Face database found")
        else:
            info.append("✗ Face database missing (run Add New User or Rebuild Database)")
        
        # Count known users
        if os.path.exists('faces'):
            users = [d for d in os.listdir('faces') if os.path.isdir(os.path.join('faces', d))]
            info.append(f"👤 Known users: {len(users)}")
        else:
            info.append("👤 Known users: 0")
        
        # Count stranger photos
        if os.path.exists('strangers'):
            strangers = len([f for f in os.listdir('strangers') if f.endswith('.jpg')])
            info.append(f"📸 Stranger photos: {strangers}")
        else:
            info.append("📸 Stranger photos: 0")
        
        # Python version
        info.append(f"🐍 Python: {sys.version.split()[0]}")
        
        messagebox.showinfo("System Information", "\n".join(info))
        self.log("System info displayed")
    
    def on_closing(self):
        """Handle window close event"""
        if self.is_running:
            if messagebox.askokcancel("Quit", "System is running. Stop and quit?"):
                self.stop_system()
                self.root.destroy()
        else:
            self.root.destroy()

def main():
    root = tk.Tk()
    app = SecuritySystemGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
