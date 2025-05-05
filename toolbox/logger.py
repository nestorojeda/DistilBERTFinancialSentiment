"""
Logger module for tracking execution logs and timing information.
"""
import time
import datetime
import os
from typing import Dict, List, Any, Optional

class Logger:
    """
    Logger class to capture execution logs and timing information.
    Logs are stored in memory during execution and can be dumped to a file at the end.
    """
    def __init__(self, log_dir: str = "../logs", filename: Optional[str] = None):
        """
        Initialize the logger with a log directory.
        
        Args:
            log_dir (str): Directory where log files will be stored
            filename (str, optional): Custom filename for the log file. If None, a timestamp will be used.
        """
        self.logs: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.timers: Dict[str, float] = {}
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Set log filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = filename or f"execution_log_{timestamp}.txt"
        self.log_path = os.path.join(log_dir, self.filename)
        
        # Log the start of execution
        self.log("Logger initialized", type="INFO")
        
    def log(self, message: str, type: str = "INFO", **kwargs):
        """
        Log a message with additional metadata.
        
        Args:
            message (str): The log message
            type (str): Log level/type (INFO, WARNING, ERROR, etc.)
            **kwargs: Additional metadata to include with the log
        """
        timestamp = time.time()
        elapsed = timestamp - self.start_time
        
        log_entry = {
            "timestamp": timestamp,
            "datetime": datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": elapsed,
            "type": type,
            "message": message
        }
        
        # Add any additional info
        log_entry.update(kwargs)
        
        # Add to logs list
        self.logs.append(log_entry)
        
        # Print to console as well
        time_str = datetime.datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
        print(f"[{time_str}] [{type}] {message}")
    
    def start_timer(self, name: str):
        """Start a named timer"""
        self.timers[name] = time.time()
        self.log(f"Timer '{name}' started", type="TIMER")
    
    def end_timer(self, name: str) -> float:
        """End a named timer and return elapsed time"""
        if name not in self.timers:
            self.log(f"Timer '{name}' does not exist", type="WARNING")
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        self.log(f"Timer '{name}' ended after {elapsed:.2f} seconds", 
                type="TIMER", 
                timer_name=name, 
                elapsed_seconds=elapsed)
        return elapsed
    
    def dump_to_file(self):
        """Write all logs to a file"""
        try:
            with open(self.log_path, 'w') as f:
                f.write(f"=== Execution Log ===\n")
                f.write(f"Start time: {datetime.datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total execution time: {time.time() - self.start_time:.2f} seconds\n\n")
                
                for log in self.logs:
                    time_str = datetime.datetime.fromtimestamp(log['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{time_str}] [{log['type']}] {log['message']}\n")
                    # Write additional metadata if present
                    for key, value in log.items():
                        if key not in ['timestamp', 'datetime', 'type', 'message']:
                            f.write(f"  - {key}: {value}\n")
            
            self.log(f"Logs dumped to {self.log_path}", type="INFO")
            return self.log_path
        except Exception as e:
            self.log(f"Error dumping logs to file: {str(e)}", type="ERROR")
            return None
            
    def get_execution_time(self) -> float:
        """Get total execution time in seconds"""
        return time.time() - self.start_time