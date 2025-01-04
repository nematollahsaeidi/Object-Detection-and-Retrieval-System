def build_message(identity, status_code, msg, exception=None):
    message = {"request_identity": identity, "status_code": status_code, "message": msg,
               "exception": exception}
    return message
