import datetime
import json
import logging
import sys
from logging.handlers import RotatingFileHandler

import json_log_formatter


class CustomisedJSONFormatter(json_log_formatter.JSONFormatter):
    def json_record(self, message, extra, record):
        msg = json.loads(message)
        extra['request_identity'] = msg['request_identity']
        extra['status_code'] = msg['status_code']
        extra['message'] = msg['message']
        if msg.get('exception') is not None:
            extra['exception'] = msg['exception']
        if 'time' not in extra:
            extra['time'] = datetime.datetime.now()
        if record.exc_info:
            extra['exc_info'] = self.formatException(record.exc_info)
        return extra


class Logger:
    def __init__(self, log_file_name):
        self.formatter = logging.Formatter("%(process)d - %(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.log_file = log_file_name

    def __get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler

    def __get_file_handler(self):
        file_handler = RotatingFileHandler(self.log_file, maxBytes=50000, backupCount=4)
        file_handler.setFormatter(self.formatter)
        return file_handler

    def __get_json_appender_handler(self):
        formatter = CustomisedJSONFormatter()
        json_handler = RotatingFileHandler(self.log_file + '.json', maxBytes=50000000, backupCount=4)
        json_handler.setFormatter(formatter)
        return json_handler

    def get_logger(self, logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(self.__get_console_handler())
        logger.addHandler(self.__get_json_appender_handler())
        logger.propagate = False
        return logger
