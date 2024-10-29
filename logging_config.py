import logging
import logging.config

def setup_logging():
    """Set up logging configuration."""
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': logging.INFO,
                'stream': 'ext://sys.stdout',
            },
        },
        'root': {
            'handlers': ['console'],
            'level': logging.INFO,
        },
    })

    logging.getLogger("py4j").setLevel(logging.ERROR)
    logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)