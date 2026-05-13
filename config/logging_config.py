import sys
import os
from loguru import logger

def setup_logging(config):
    """
    Configures Loguru for high-performance trading.
    Prevents terminal crashes by limiting console output while keeping detailed logs in files.
    """
    # 1. Remove default Loguru handler
    logger.remove()

    # 2. Add a DETAILED file handler (Rotation: 100MB or 1 day, Retention: 10 days)
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "trading_bot.log")
    
    # Advanced rotation disabled on Windows to prevent PermissionError [WinError 32]
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module}:{function}:{line} - {message}",
        backtrace=True,
        diagnose=True
    )

    # 3. Add a CRITICAL ERROR file handler
    error_log = os.path.join(log_dir, "critical_errors.log")
    logger.add(
        error_log,
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}"
    )

    # 4. Add a CLEAN CONSOLE handler (Higher level to prevent terminal overflow)
    # We only show SUCCESS and above by default to keep terminal clean
    console_level = config.get("logging", {}).get("console_level", "SUCCESS")
    
    logger.add(
        sys.stderr,
        level=console_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        colorize=True
    )

    # 5. Intercept standard logging (from libraries like SmartApi, websockets, etc.)
    import logging
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # 6. Specifically silence extremely noisy libraries if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("SmartApi").setLevel(logging.WARNING)
    logging.getLogger("_logging").setLevel(logging.WARNING) # Silence internal Angel One ping/pong logs

    logger.success("🛡️ LOGGING SYSTEM INITIALIZED — FULL PROOF MODE ACTIVE")
