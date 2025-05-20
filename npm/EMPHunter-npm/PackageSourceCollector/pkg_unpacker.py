import os
import filetype
import random
import requests
# 遍历文件夹及其子文件夹中的文件，并存储在一个列表中

# 输入文件夹路径、空文件列表[]

# 返回 文件列表Filelist,包含文件名（完整路径）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_file_list(file_list, suffix, target_dir):
    for i in os.listdir(target_dir):
        if os.path.isfile(os.path.join(target_dir, i)) and i.endswith(suffix):
            file_list.append(os.path.join(target_dir, i))
        elif os.path.isdir(os.path.join(target_dir, i)):
            file_list = get_file_list(file_list, suffix, os.path.join(target_dir, i))
    return file_list


def unpackage_npm(batch_name):
    package_dir = os.path.join(CURRENT_DIR, "../../Packages/packages", batch_name)
    unpacked_dir = os.path.join(CURRENT_DIR, "../../Packages/unpacked", batch_name)
    if not os.path.exists(CURRENT_DIR + "/../../Packages/unpacked"):
        os.mkdir(CURRENT_DIR + "/../../Packages/unpacked")
    if not os.path.exists(unpacked_dir):
        os.mkdir(unpacked_dir)

    list = get_file_list([], ".tgz", package_dir)
    for e in list:
        kind = filetype.guess(e)
        pkg_name = ".".join(e.split("/")[-1].split(".")[:-1])
        if kind is None:
            print('Cannot guess file type!')
            continue
        else:
            extension = kind.extension
            if extension == "gz":
                if not os.path.exists(os.path.join(unpacked_dir, pkg_name)):
                    os.mkdir(os.path.join(unpacked_dir, pkg_name))
                cmd = "tar zxvf " + e + " -C " + os.path.join(unpacked_dir, pkg_name)
            else:
                cmd = ""
            print(cmd)
            os.system(cmd)


def unpack_4_train(target_dir):
    file_list = get_file_list([], ".tgz", target_dir)
    unpacked_dir = os.path.join(CURRENT_DIR, "../../Packages/js_train_sample")

    for i in file_list:
        pkg_name = ".".join(i.split("/")[-1].split(".")[:-1])
        if not os.path.exists(os.path.join(unpacked_dir, pkg_name)):
            os.mkdir(os.path.join(unpacked_dir, pkg_name))
            cmd = "tar zxvf " + i + " -C " + os.path.join(unpacked_dir, pkg_name)
            os.system(cmd)


def unpack_4_any(target_dir, unpacked_dir):
    file_list = get_file_list([], ".tgz", target_dir)
    print(file_list)

    for file_path in file_list:
        pkg_name = ".".join(file_path.split("/")[-1].split(".")[:-1])
        if not os.path.exists(os.path.join(unpacked_dir, pkg_name)):
            os.mkdir(os.path.join(unpacked_dir, pkg_name))
        cmd = "tar zxvf " + file_path + " -C " + os.path.join(unpacked_dir, pkg_name)
        os.system(cmd)


def unpack_4_a_day(target_dir_dir, unpacked_dir, date):
    if not os.path.exists(unpacked_dir):
        os.mkdir(unpacked_dir)
    tmp_dir_list = os.listdir(target_dir_dir)
    dir_list = []
    for i in tmp_dir_list:
        if i.startswith(date):
            dir_list.append(i)
    for target_dir in dir_list:
        target_dir = os.path.join(target_dir_dir, target_dir)
        file_list = get_file_list([], ".tgz", target_dir)
        # print(file_list)

        for file_path in file_list:
            pkg_name = ".".join(file_path.split("/")[-1].split(".")[:-1])
            if not os.path.exists(os.path.join(unpacked_dir, pkg_name)):
                os.mkdir(os.path.join(unpacked_dir, pkg_name))
            cmd = "tar zxvf " + file_path + " -C " + os.path.join(unpacked_dir, pkg_name)
            os.system(cmd)


def unpack_4_random(target_dir, unpacked_dir, limit):
    all_file_list = get_file_list([], ".tgz", target_dir)
    # print(file_list)
    import random
    file_list = random.sample(all_file_list, limit)

    for file_path in file_list:
        pkg_name = ".".join(file_path.split("/")[-1].split(".")[:-1])
        if not os.path.exists(os.path.join(unpacked_dir, pkg_name)):
            os.mkdir(os.path.join(unpacked_dir, pkg_name))
        cmd = "tar zxvf " + file_path + " -C " + os.path.join(unpacked_dir, pkg_name)
        os.system(cmd)


def unpack_4_benign_random(target_dir, unpacked_dir, limit):
    all_file_list = get_file_list([], ".tgz", target_dir)
    # print(file_list)
    import random
    file_list = random.sample(all_file_list, limit)

    for file_path in file_list:
        pkg_name = ".".join(file_path.split("/")[-1].split(".")[:-1])
        if not os.path.exists(os.path.join(unpacked_dir, pkg_name)):
            os.mkdir(os.path.join(unpacked_dir, pkg_name))
        cmd = "tar zxvf " + file_path + " -C " + os.path.join(unpacked_dir, pkg_name)
        os.system(cmd)


def select_4_random(target_dir, unpacked_dir, limit):
    all_file_list = os.listdir(target_dir)
    for i in all_file_list:
        print(i)
    # print(file_list)
    # import random
    # file_list = random.sample(all_file_list, limit)
    #
    # for file_path in file_list:
    #     pkg_name = file_path
    #     file_path = os.path.join(target_dir, pkg_name)
    #     # if not os.path.exists(os.path.join(unpacked_dir, pkg_name)):
    #     #     os.mkdir(os.path.join(unpacked_dir, pkg_name))
    #     cmd = "mv -i " + file_path + " " + unpacked_dir
    #     os.system(cmd)


def get_name_list(target_dir_dir, unpacked_dir, date, num):
    if not os.path.exists(unpacked_dir):
        os.mkdir(unpacked_dir)
    tmp_dir_list = os.listdir(target_dir_dir)
    dir_list = []
    for i in tmp_dir_list:
        if i.startswith(date):
            dir_list.append(i)
    pkg_list = []
    for target_dir in dir_list:
        target_dir = os.path.join(target_dir_dir, target_dir)
        file_list = get_file_list([], ".tgz", target_dir)
        # print(file_list)

        for file_path in file_list:
            pkg_name = ".".join(file_path.split("/")[-1].split(".")[:-1])
            pkg_list.append(pkg_name)

    with open(os.path.join(CURRENT_DIR, "../../NPM/DataSet/Target-Batch", num + ".txt"), "w+") as fp:
        for i in pkg_list:
            fp.write(i + "\n")


if __name__ == '__main__':
    # unpackage_npm("20240119-1")
    target_dir = os.path.join(r"/home/liang/Desktop/NPM-MPHunter/malicious/Backstabbers-Knife-Collection-master/Backstabbers-Knife-Collection-master/samples/npm")
    target_dir = os.path.join("/media/liang/Seagate-Back-up-Plus-Drive/TaoTaoBak-ThinkPad-Ubuntu/NPM-MPHunter/Packages/packages/")
    # target_dir = os.path.join(CURRENT_DIR, "../../Packages/packages/")
    # target_dir = os.path.join(CURRENT_DIR, "../../Packages/malicious_sample/20240323-Pa")
    # target_dir = os.path.join(CURRENT_DIR, "../../Packages/malicious_sample/Pa-10")
    date = "20240202"
    # num = "30"
    # unpacked_dir = os.path.join(CURRENT_DIR, "../../Packages/malicious_sample/20240711-bakset-Pa")
    # unpacked_dir = os.path.join(CURRENT_DIR, "../../Packages/benign_sample/20240711-2")
    unpacked_dir = os.path.join(CURRENT_DIR, "../../Packages/unpacked/", "canonical-" + date + "-a")
    # unpacked_dir = os.path.join("/media/liang/MyPassport/TaoTaoBak/ProjectBak/NPM-MPHunter/Packages/unpacked", "canonical-" + date + "-a")
    # unpacked_dir = os.path.join(CURRENT_DIR, "../../Packages/train_sample/20240702-1")
    # unpack_4_any(target_dir, unpacked_dir)
    # target_dir = os.path.join(CURRENT_DIR, "../../Packages/packages")
    unpack_4_a_day(target_dir, unpacked_dir, date)
    # get_name_list(target_dir, unpacked_dir, date, num)
    # unpack_4_random(target_dir, unpacked_dir, 200)
    # select_4_random(target_dir, unpacked_dir, 10)
    # unpackage_npm("20240402-15")
    # unpackage_npm("20240403-2")
    # unpackage_npm("20240403-14")
    # unpackage_npm("20240405-17")
    # unpackage_npm("20240405-19")
    # unpackage_npm("20240406-35")
    # date = "20240618"
    # unpacked_dir = os.path.join("/media/liang/MyPassport/TaoTaoBak/ProjectBak/NPM-MPHunter/Packages/unpacked", "canonical-" + date + "-a")
    # unpack_4_a_day(target_dir, unpacked_dir, date)
    # date = "20240619"
    # unpacked_dir = os.path.join("/media/liang/MyPassport/TaoTaoBak/ProjectBak/NPM-MPHunter/Packages/unpacked", "canonical-" + date + "-a")
    # unpack_4_a_day(target_dir, unpacked_dir, date)
    # date = "20240406"
    # unpacked_dir = os.path.join("/media/liang/MyPassport/TaoTaoBak/ProjectBak/NPM-MPHunter/Packages/unpacked", "canonical-" + date + "-a")
    # unpack_4_a_day(target_dir, unpacked_dir, date)
    # date = "20240405"
    # unpacked_dir = os.path.join("/media/liang/MyPassport/TaoTaoBak/ProjectBak/NPM-MPHunter/Packages/unpacked", "canonical-" + date + "-a")
    # unpack_4_a_day(target_dir, unpacked_dir, dat