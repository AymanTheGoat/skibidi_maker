import sys
from datetime import datetime

class Logger:
    # ANSI color codes
    COLORS = {
        'INFO': '\033[94m',      # Blue
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'RESET': '\033[0m'       # Reset to default
    }
    
    def __init__(self):
        pass
    
    def _log(self, level, message):
        """Internal method to format and print log messages"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = self.COLORS.get(level, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        formatted_message = f"{color}[{level}]{reset} - {message}"
        print(formatted_message)
        
        # # Also write to file without colors
        # try:
        #     with open('skibidi_log.txt', 'a', encoding='utf-8') as f:
        #         f.write(f"[{timestamp}] [{level}] - {message}\n")
        # except:
        #     pass  # Don't fail if we can't write to log file
    
    def info(self, message):
        """Log info message in blue"""
        self._log('INFO', message)
    
    def warning(self, message):
        """Log warning message in yellow"""
        self._log('WARNING', message)
    
    def error(self, message):
        """Log error message in red"""
        self._log('ERROR', message)
        exit(1) # Exit because no sane person going to continue processing after an error