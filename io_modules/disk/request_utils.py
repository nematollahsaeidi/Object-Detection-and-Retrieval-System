import datetime
import os
import shutil
import uuid
import configparser

config = configparser.ConfigParser()
config.read('cfg/config.cfg')


def initiate_request(req):
    request_time_string = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S-%f")
    request_id = uuid.uuid4().hex
    ip = req.remote_addr
    request_identity = ip + " - " + request_id + " "
    return request_identity, request_time_string


def check_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)


class RequestDirectoryManager:
    def __init__(self, base_dir=config.get('dirs', 'request_dir')):
        self.all_request_videos_dir = base_dir

    def check_for_base_directory(self):
        check_dir(self.all_request_videos_dir)

    def create_dir_for_request(self, request_time_string):
        single_request_dir = self.get_request_dir_name(request_time_string)
        self.check_for_base_directory()
        check_dir(single_request_dir)
        return single_request_dir

    def delete_dir_for_request(self, request_time_string):
        single_request_dir = self.get_request_dir_name(request_time_string)
        shutil.rmtree(single_request_dir)
        return single_request_dir

    def get_request_dir_name(self, request_time_string):
        return self.all_request_videos_dir + request_time_string + os.path.sep
