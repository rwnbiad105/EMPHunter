import json
import random
import os, re, subprocess
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED, as_completed
import threading
from func_timeout import func_set_timeout
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# ################# for threadpool #########################################
import queue
from concurrent.futures import ThreadPoolExecutor


class BoundThreadPoolExecutor(ThreadPoolExecutor):

    def __init__(self, *args, **kwargs):
        super(BoundThreadPoolExecutor, self).__init__(*args, **kwargs)
        self._work_queue = queue.Queue(14)
#############################################################################


# @func_set_timeout(30)
def download_single(pkg_name, package_dir):
    cmd = "cd " + package_dir + "; npm pack " + pkg_name
    os.system(cmd)
    return pkg_name


def download_npm_in_list(batch_name):
    package_dir = os.path.join(CURRENT_DIR, "../../Packages/packages", batch_name)

    with open(os.path.join(package_dir, batch_name + "_names.txt"), "r") as f:
        name_list = f.readlines()

    with ThreadPoolExecutor(max_workers=14) as pool:
        obj_list = []
        for pkg_name in name_list:
            if pkg_name == "":
                continue
            obj = pool.submit(lambda cpx: download_single(cpx, package_dir), (pkg_name))
            obj_list.append(obj)
        count = 0
        for future in as_completed(obj_list):
            count += 1
            try:
                print(str(count) + "done:" + future.result())
            except:
                pass


if __name__ == "__main__":
    download_npm_in_list("20240119-1")
