import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Set up logging configuration."""
    # Clean up the log level string and handle comments
    log_level = log_level.split('#')[0].strip().upper()
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)s | %(message)s | %(extra_data)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s     | %(message)s'
    )
    
    # Set up console handler with UTF-8 encoding
    if sys.platform == 'win32':
        # On Windows, use sys.stdout with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
    else:
        # On other platforms, use regular StreamHandler
        console_handler = logging.StreamHandler()
    
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # Set up file handler if log file is specified
    if log_file:
        # Use UTF-8 encoding for file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)

class ExtraDataAdapter(logging.LoggerAdapter):
    """
    Adapter to add extra data to log records in a structured way.
    """
    def process(self, msg, kwargs):
        # Ensure extra_data exists in kwargs
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        if 'extra_data' not in kwargs['extra']:
            kwargs['extra']['extra_data'] = {}
            
        # If extra data was passed directly to the log call, merge it
        if 'extra_data' in kwargs:
            kwargs['extra']['extra_data'].update(kwargs.pop('extra_data'))
            
        # Handle Unicode characters in extra_data
        try:
            # Convert extra_data to string, handling Unicode
            if isinstance(kwargs['extra']['extra_data'], dict):
                # For dictionaries, handle each value separately
                cleaned_dict = {}
                for k, v in kwargs['extra']['extra_data'].items():
                    if isinstance(v, str):
                        # Replace problematic Unicode characters with ASCII equivalents
                        v = v.replace('✓', 'YES').replace('✗', 'NO')
                    cleaned_dict[k] = v
                kwargs['extra']['extra_data'] = str(cleaned_dict)
            else:
                # For non-dictionaries, convert to string and handle Unicode
                extra_data_str = str(kwargs['extra']['extra_data'])
                extra_data_str = extra_data_str.replace('✓', 'YES').replace('✗', 'NO')
                kwargs['extra']['extra_data'] = extra_data_str
        except Exception as e:
            kwargs['extra']['extra_data'] = f"Error converting extra_data: {str(e)}"
        
        return msg, kwargs

def get_logger(name: str) -> ExtraDataAdapter:
    """
    Get a logger instance with the ExtraDataAdapter.
    
    Args:
        name: Logger name, typically __name__ of the module
        
    Returns:
        Configured logger instance with ExtraDataAdapter
    """
    logger = logging.getLogger(f'travel_advisor.{name}')
    return ExtraDataAdapter(logger, {}) 