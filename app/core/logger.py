import logging
import asyncio
import inspect
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import sys

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    
    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_BLUE = '\033[104m'
    BG_MAGENTA = '\033[105m'
    BG_CYAN = '\033[106m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    
    # Special symbols
    SUCCESS = '✅'
    ERROR = '❌'
    WARNING = '⚠️'
    INFO = 'ℹ️'
    DEBUG = '🔍'
    CRITICAL = '💥'
    ARROW = '➤'
    SEPARATOR = '─' * 50


class RichAsyncFormatter(logging.Formatter):
    """Custom formatter with rich colors and detailed context information"""
    
    def __init__(self):
        super().__init__()
        
        self.level_colors = {
            'DEBUG': Colors.CYAN,
            'INFO': Colors.GREEN,
            'WARNING': Colors.YELLOW,
            'ERROR': Colors.RED,
            'CRITICAL': Colors.BG_RED + Colors.WHITE + Colors.BOLD,
        }
        
        self.level_symbols = {
            'DEBUG': Colors.DEBUG,
            'INFO': Colors.INFO,
            'WARNING': Colors.WARNING,
            'ERROR': Colors.ERROR,
            'CRITICAL': Colors.CRITICAL,
        }

    def format(self, record):
        frame = inspect.currentframe()
        try:
            caller_frame = frame
            
            while caller_frame:
                filename = caller_frame.f_code.co_filename
             
                if not any(log_file in filename for log_file in ['logging', __file__]):
                    break
                caller_frame = caller_frame.f_back

            if caller_frame:
                file_path = Path(caller_frame.f_code.co_filename)
                line_number = caller_frame.f_lineno
                function_name = caller_frame.f_code.co_name
            else:
                file_path = Path(record.pathname)
                line_number = record.lineno
                function_name = record.funcName

        except Exception:
            file_path = Path(record.pathname)
            line_number = record.lineno
            function_name = record.funcName
        finally:
            del frame

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        level_color = self.level_colors.get(record.levelname, Colors.WHITE)
        level_symbol = self.level_symbols.get(record.levelname, '•')
        
        async_info = ""
        try:
            loop = asyncio.get_running_loop()
            current_task = asyncio.current_task(loop)
            if current_task:
                task_name = getattr(current_task, 'get_name', lambda: 'Unknown')()
                async_info = f"{Colors.MAGENTA}[Task: {task_name}]{Colors.RESET} "
        except RuntimeError:
            pass

        formatted_parts = [
            f"{Colors.DIM}{timestamp}{Colors.RESET}",
            f"{level_symbol} {level_color}{Colors.BOLD}[{record.levelname:>8}]{Colors.RESET}",
    
            async_info,
            
            f"{Colors.BLUE}{Colors.UNDERLINE}{file_path.name}:{line_number}{Colors.RESET}",
            
            f"{Colors.CYAN}in {function_name}(){Colors.RESET}",
            
            f"{Colors.DIM}{Colors.ARROW}{Colors.RESET}",
            
            f"{Colors.WHITE}{record.getMessage()}{Colors.RESET}"
        ]
        
        formatted_message = " ".join(formatted_parts)
        
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            formatted_message += f"\n{Colors.RED}{exc_text}{Colors.RESET}"
            
        return formatted_message

    def formatException(self, ei):
        """Format exception with colors"""
        lines = traceback.format_exception(*ei)
        colored_lines = []
        
        for line in lines:
            if line.strip().startswith('File'):
                # Color file paths
                colored_lines.append(f"{Colors.BLUE}{line.rstrip()}{Colors.RESET}")
            elif line.strip().startswith(('Traceback', 'During handling')):
                # Color traceback headers
                colored_lines.append(f"{Colors.YELLOW}{line.rstrip()}{Colors.RESET}")
            elif line.strip() and not line.startswith(' '):
                # Color exception names
                colored_lines.append(f"{Colors.RED}{Colors.BOLD}{line.rstrip()}{Colors.RESET}")
            else:
                # Regular code lines
                colored_lines.append(f"{Colors.WHITE}{line.rstrip()}{Colors.RESET}")
                
        return '\n'.join(colored_lines)


class RichAsyncLogger:
    """Rich logger with async support and detailed context"""
    
    def __init__(self, name: str | None = None, level: int = logging.DEBUG):
        self.name = name or self._get_caller_module()
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create console handler with rich formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(RichAsyncFormatter())
        
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to avoid duplicate messages
        self.logger.propagate = False
    
    def _get_caller_module(self) -> str:
        """Get the module name of the caller"""
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back.f_back  # Skip this method and __init__
            if caller_frame:
                module = inspect.getmodule(caller_frame)
                return module.__name__ if module else 'unknown'
            return 'unknown'
        finally:
            del frame

    def debug(self, message: Any, *args, **kwargs):
        """Log debug message"""
        self.logger.debug(str(message), *args, **kwargs)
    
    def info(self, message: Any, *args, **kwargs):
        """Log info message"""
        self.logger.info(str(message), *args, **kwargs)
    
    def warning(self, message: Any, *args, **kwargs):
        """Log warning message"""
        self.logger.warning(str(message), *args, **kwargs)
    
    def warn(self, message: Any, *args, **kwargs):
        """Alias for warning"""
        self.warning(message, *args, **kwargs)
    
    def error(self, message: Any, *args, **kwargs):
        """Log error message"""
        self.logger.error(str(message), *args, **kwargs)
    
    def critical(self, message: Any, *args, **kwargs):
        """Log critical message"""
        self.logger.critical(str(message), *args, **kwargs)
    
    def exception(self, message: Any, *args, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(str(message), *args, **kwargs)
    
    def success(self, message: Any, *args, **kwargs):
        """Log success message (custom level)"""
        self.info(f"{Colors.SUCCESS} {message}", *args, **kwargs)
    
    def separator(self, message: str = ""):
        """Print a separator line"""
        sep_msg = f"{Colors.DIM}{Colors.SEPARATOR}"
        if message:
            sep_msg += f" {message} {Colors.SEPARATOR}"
        sep_msg += Colors.RESET
        print(sep_msg)
    
    def header(self, message: str):
        """Print a header message"""
        header_msg = f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 20} {message} {'=' * 20}{Colors.RESET}\n"
        print(header_msg)


def get_rich_logger(name: str | None = None  , level: int = logging.DEBUG) -> RichAsyncLogger:
    """Create and return a rich async logger"""
    return RichAsyncLogger(name, level)



if __name__ == "__main__":
    logger = get_rich_logger("demo")
    
    async def demo_async_function():
        """Demo function showing logger in async context"""
        logger.info("Starting async demo function")
        
        await asyncio.sleep(0.1)
        logger.debug("After small delay")
        
        logger.success("Async operation completed successfully!")
        
        try:
            result = 10 / 0
        except Exception as e:
            logger.exception("An error occurred in async function")
    
    def demo_sync_function():
        """Demo function showing logger in sync context"""
        logger.info("Starting sync demo function")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        logger.critical("This is a critical message")
    
    async def main():
        """Main demo function"""
        logger.header("RICH ASYNC LOGGER DEMO")
        
        logger.info("Logger initialized successfully")
        logger.debug("This is a debug message with detailed info")
        
        logger.separator("SYNC DEMO")
        demo_sync_function()
        
        logger.separator("ASYNC DEMO")
        
        tasks = [
            asyncio.create_task(demo_async_function(), name=f"DemoTask-{i}")
            for i in range(2)
        ]
        
        await asyncio.gather(*tasks)
        
        logger.success("All demo tasks completed!")
        logger.separator()
    
    asyncio.run(main())