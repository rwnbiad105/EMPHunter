import json, os, random, requests, datetime, pickle, re, dns.resolver
from urllib import parse as url_parse
import multiprocessing
import numpy as np
from gensim.models import FastText
from gensim.models import Word2Vec
from concurrent.futures import ThreadPoolExecutor, as_completed
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class TrainParser:
    def __init__(self, is_canonical=True, cmd_seq_dict={}, js_seq_dict={}, pkg_name_2_file_path_dict={},
                 batch_name="test", pkg_id_2_pkg_name_dict={}, is_original=False):
        self.is_original = is_original
        self.is_canonical = is_canonical
        self.cmd_seq_dict = cmd_seq_dict
        self.js_seq_dict = js_seq_dict
        self.pkg_name_2_file_path_dict = pkg_name_2_file_path_dict
        self.pkg_id_2_pkg_name_dict = pkg_id_2_pkg_name_dict
        self.batch_name = batch_name

        self.is_e_js = False
        self.is_c_bash = False
        self.previous_is_node = False
        self.previous_is_bash = False
        self.is_call = False
        self.previous_is_identifier = False
        self.special_list = ["&", "&&", "||", "|"]
        self.ignore_list = ["$", "$(", "*", "(", ")", "[", "]", "{", "}", ",", ";", ":", "?", ".", "<", ">", "~", "`", "\"", "\'", "\\", "\\\"", "\\\'"]
        self.js_pkg_id = -0.5

        self.cfg_def_func_dict = {}
        self.bb_list = []
        self.func_def_dict = {}
        self.func_use_dict = {}
        self.if_count = 0
        self.js_lv = 0
        self.bash_lv = 0
        self.cmd_pkg_ex_id = 0
        self.js_wv_model = None
        self.cmd_wv_model = None
        self.cmd_centroid_vec = None
        self.cmd_substitution_token = ""
        self.raw_cmd = ""

    def parse(self, code, language):
        from tree_sitter import Language, Parser
        LANGUAGE = Language(os.path.join(CURRENT_DIR, './treesitter/build/languages.so'), language)

        # 举一个bash例子
        parser = Parser()
        parser.set_language(LANGUAGE)

        # 没报错就是成功
        try:
            tree = parser.parse(bytes(code, "utf8"))
            root_node = tree.root_node
        except:
            root_node = None
            print("parsing fail:")
            print(code)
        # 注意，root_node 才是可遍历的树节点

        return root_node

    def traverse_js_ast(self, root_node, is_large_file, pkg_id, we_train_batch_name = None, centroid_vec=None):
        def get_ret(root_node):
            # 用于在遍历到一个树的叶子节点后返回结果，存到canonical list中，以path.join举例
            content = root_node.text.decode("utf-8")
            if root_node.type == "." and self.is_call:
                # 返回“.”
                self.previous_is_identifier = False
                return content
            elif (root_node.type == "identifier" or root_node.type == "property_identifier") and self.is_call:
                if self.previous_is_identifier:
                    if content in self.func_def_dict.keys():
                        if len(self.func_def_dict[content]) < 2000:
                            content = self.func_def_dict[content]
                            self.func_use_dict[content] = True
                    # 是函数库名，返回path
                    return " " + content
                else:
                    # 是函数方法名，即，“.”后面的函数名，返回join
                    self.previous_is_identifier = True
                    return content
            else:
                self.is_call = False
                return ""

        def canonical_sort(centroid_vec):
            def cosine_similarity(vec1, vec2):
                # 计算两个向量的内积
                dot_product = np.dot(vec1, vec2)
                # 计算两个向量的范数
                norm_vec1 = np.linalg.norm(vec1)
                norm_vec2 = np.linalg.norm(vec2)
                # 计算余弦相似度
                cosine_sim = dot_product / (norm_vec1 * norm_vec2)
                return cosine_sim

            dist_dict = {}
            # print(self.bb_list)
            for bb in self.bb_list:
                if len(bb.strip().split()):
                    first_word = bb.strip().split()[0]
                    vec = self.js_wv_model.wv[first_word]
                    dist_dict[bb] = cosine_similarity(vec, centroid_vec)

            self.bb_list = sorted(dist_dict, key=dist_dict.get)
            # print("#", self.bb_list)

        def attach_unused_def_func(content):
            if root_node.type == "program":
                for func_name, is_used in self.func_use_dict.items():
                    if not is_used:
                        content += " " + self.func_def_dict[func_name].strip()
            return content

        # ########### start to parse ########### #
        # if self.js_lv >= 35:
        #     return ""
        self.js_lv += 1
        if centroid_vec is None and self.is_canonical:
            with open(os.path.join(CURRENT_DIR, "../DataShare/Model/WordEmbedding", we_train_batch_name,
                                   "js_centroid_vec.pkl"), "rb+") as fp:
                centroid_vec = pickle.load(fp)

        if len(root_node.children):
            # print("  " * (self.js_lv - 1), root_node.type)
            # 如果有子节点就继续dfs
            code_tokens = ""
            if root_node.type == "call_expression" or root_node.type == "function_expression":
                self.is_call = True
            elif not root_node.type == "member_expression":
                self.is_call = False

            if not (root_node.type == "if_statement"
                    or root_node.type == "for_statement"
                    or root_node.type == "while_statement"
                    or root_node.type == "try_statement"
                    or root_node.type == "switch_statement"
                    or root_node.type == "function_declaration"):
                for child in root_node.children:
                    child_tokens = self.traverse_js_ast(child, is_large_file, pkg_id, we_train_batch_name, centroid_vec)
                    code_tokens += child_tokens
            elif root_node.type == "function_declaration":
                content = root_node.text.decode("UTF-8")
                index = content.find("function ")
                if index != -1:
                    # 获取子串后面的内容
                    func_name = content[index + len("function "):].split("(")[0].strip()
                    # func_name = content.strip("function").split("(")[0].strip()
                    def_tokens = ""
                    for child in root_node.children:
                        child_tokens = self.traverse_js_ast(child, is_large_file, pkg_id, we_train_batch_name,
                                                            centroid_vec)
                        def_tokens += child_tokens
                    if def_tokens.strip() != "":
                        self.is_call = False
                        self.func_def_dict[func_name] = def_tokens
                        self.func_use_dict[func_name] = False
            else:
                temp_bb_words_list = []
                for child in root_node.children:
                    child_tokens = self.traverse_js_ast(child, is_large_file, pkg_id, we_train_batch_name,
                                                        centroid_vec)
                    if child.type == "else_clause" \
                            or child.type == "catch_clause" \
                            or child.type == "finally_clause" \
                            or child.type == "switch_default" \
                            or child.type == "switch_case":
                        self.bb_list.append("".join(temp_bb_words_list))
                        temp_bb_words_list = []

                    if len(child_tokens):
                        # print(child_tokens)
                        temp_bb_words_list.append(child_tokens)
                if len(temp_bb_words_list):
                    self.bb_list.append("".join(temp_bb_words_list))
                if self.is_canonical:
                    canonical_sort(centroid_vec)
                code_tokens += " ".join(self.bb_list)
                # print(temp_bb_words_list)
                # print(self.bb_list)
                self.bb_list = []
            # print(code_tokens)

            if root_node.type == "program" and self.js_lv == 1 and not is_large_file:
                attach_unused_def_func(code_tokens)
            self.js_lv -= 1
            return code_tokens
        else:
            ret = get_ret(root_node)
            self.js_lv -= 1
            return ret

    def traverse_bash_ast(self, root_node, pkg_id, we_train_batch_name=None, parents_type=None, command_name=None, is_original=False):
        self.bash_lv += 1
        # print(" "*self.bash_lv, root_node.type, root_node.text.decode("utf-8"), parents_type)

        if is_original:
            content = root_node.text.decode("utf-8")
            if root_node.type == "program":
                self.cmd_seq_dict[pkg_id] = content
                self.raw_cmd = content

            if len(root_node.children):
                self.previous_is_bash = False
                self.previous_is_node = False
                if root_node.text.decode("utf-8") == "node":
                    self.previous_is_node = True
                elif root_node.text.decode("utf-8") == "bash":
                    self.previous_is_bash = True

                for child in root_node.children:
                    self.traverse_bash_ast(child, pkg_id, we_train_batch_name, root_node.type, command_name, is_original)
            else:
                # leaf node
                path = os.path.join(self.pkg_name_2_file_path_dict[self.pkg_id_2_pkg_name_dict[pkg_id]],
                                    "package", content)
                if os.path.exists(path) and len(content) and content != ".":
                    if content.endswith(".sh"):
                        self.ex_sh_parser(pkg_id, None, path, we_train_batch_name, is_original)
                else:
                    if self.is_c_bash and root_node.type in ["string_content", "string", "raw_string"]:
                        print(self.raw_cmd)
                        print(content)
                        self.ex_sh_parser(pkg_id, content, None, we_train_batch_name, is_original)

                if content.startswith("-") and parents_type == "command":
                    # 如果是选项
                    self.is_c_bash = False
                    if content == "-c" and self.previous_is_bash:
                        self.is_c_bash = True
            return self.raw_cmd

        if self.cmd_centroid_vec is None and self.is_canonical:
            # 没加载base vec的话加载
            with open(os.path.join(CURRENT_DIR, "../DataShare/Model/WordEmbedding", we_train_batch_name,
                                   "cmd_centroid_vec.pkl"), "rb+") as fp:
                self.cmd_centroid_vec = pickle.load(fp)

        if root_node.type == "command":
            # 如果是命令的话，获取命令名
            command_name = root_node.text.decode("utf-8").split(" ")[0]
            path = os.path.join(self.pkg_name_2_file_path_dict[self.pkg_id_2_pkg_name_dict[pkg_id]],
                                "package", command_name)
            if os.path.exists(path) and len(command_name) and command_name != ".":
                command_name = path.split("/")[-1]

        if len(root_node.children):
            #如果有孩子分支的话
            code_tokens = ""
            parents_type = root_node.type
            self.previous_is_bash = False
            self.previous_is_node = False
            if root_node.text.decode("utf-8") == "node":
                self.previous_is_node = True
            elif root_node.text.decode("utf-8") == "bash":
                self.previous_is_bash = True

            for child in root_node.children:
                child_tokens = self.traverse_bash_ast(child, pkg_id, we_train_batch_name, parents_type, command_name, is_original=False)
                # print(" "*self.bash_lv, root_node.type, child_tokens)
                # if child_tokens == "-bash.c":
                #     pass
                # if child.type == "command":
                #     print("command:", child_tokens)
                #     if root_node.parent:
                #         print(root_node.parent.type)

                if len(child_tokens) and child.type == "command":
                    # if not child_tokens.endswith(" ;"):
                    #     child_tokens += " ;"
                    if self.is_canonical:
                        child_tokens = self.cmd_canonical_sort(child_tokens)
                    if len(self.cmd_substitution_token):
                        # print(" " * self.bash_lv, "code_tokens:", code_tokens)
                        # 当处理完一整条命令后，将嵌套的命令连接上
                        if not child_tokens.startswith(
                                (" ;", " &&", " ||")) and not self.cmd_substitution_token.endswith((";", "&&", "||")):
                            child_tokens = " ;" + child_tokens
                        child_tokens = self.cmd_substitution_token + child_tokens
                        self.cmd_substitution_token = ""
                if child.type == "command_substitution":
                    self.cmd_substitution_token += child_tokens
                    child_tokens = " command_substitution"
                if child_tokens in [" ;", " &&", " ||", " |"] and child.type == "command_substitution":
                    child_tokens = " ;"
                    self.cmd_substitution_token += child_tokens
                    child_tokens = ""
                # print(" "*self.bash_lv, "subcmd_tokens:", self.cmd_substitution_token)

                code_tokens += child_tokens

            # 当处理完一整条命令后，将嵌套的命令连接上
            if len(self.cmd_substitution_token) and root_node.type == "program":
                if not code_tokens.startswith(
                        (" ;", " &&", " ||")) and not self.cmd_substitution_token.endswith((";", "&&", "||")) :
                    code_tokens = " ;" + code_tokens
                code_tokens = self.cmd_substitution_token + code_tokens
                self.cmd_substitution_token = ""

            seq = code_tokens.strip().replace("\n", " ")
            seq = " ".join(seq.split())
            self.cmd_seq_dict[pkg_id] = seq
            self.bash_lv -= 1
            return code_tokens
        else:
            # 如果是叶子节点的话
            if parents_type == "program":
                self.bash_lv -= 1
                return ""
            else:
                ret = ""
                content = root_node.text.decode("utf-8")
                if content.startswith("-") and parents_type == "command":
                    # 如果是选项
                    self.is_e_js = False
                    self.is_c_bash = False
                    if content == "-e" and self.previous_is_node:
                        self.is_e_js = True
                    if content == "-c" and self.previous_is_bash:
                        self.is_c_bash = True
                    ret += " -" + command_name + "." + content.strip("-")
                    if ret.count("=") == 1:
                        code_list = ret.split("=")
                        content = "=".join(code_list[1:])
                        ret = code_list[0]
                    else:
                        content = ""
                        
                # 如果不算选项,而是路径的话
                path = os.path.join(self.pkg_name_2_file_path_dict[self.pkg_id_2_pkg_name_dict[pkg_id]],
                                    "package", content)
                if os.path.exists(path) and len(content) and content != ".":
                    if content.endswith("js"):
                        self.js_parser(path, None, pkg_id, we_train_batch_name)
                        ret += " js_filepath"
                    elif content.endswith(".sh"):
                        self.ex_sh_parser(pkg_id, None, path, we_train_batch_name, is_original)
                        ret += " sh_filepath"
                    elif parents_type == "command_name":
                        command_name = path.split("/")[-1]
                        ret += " " + command_name
                    else:
                        ret += " filepath"
                    self.bash_lv -= 1
                    return ret
                else:
                    # 如果是不存在的路径的话
                    path_pattern = r'^(?:(?:[a-zA-Z]:|\.{1,2})?[\\/](?:[^\\?/*|<>:"]+[\\/])*)(?:(?:[^\\?/*|<>:"]+?)(?:\.[^.\\?/*|<>:"]+)?)?$'
                    if re.match(path_pattern, content) is not None or ("/" in content and not " " in content):
                        url_ret = url_parse.urlparse(content)
                        if url_ret.scheme == "" and url_ret.netloc == "":
                            self.bash_lv -= 1
                            return " empty_filepath"
                        
                # 如果也不是路径
                ret_appdex = ""
                if len(content):
                    ret_appdex = " " + root_node.type

                if self.is_e_js and root_node.type in ["string_content", "string", "raw_string"]:
                    import string
                    striped_content = content.strip(string.punctuation)
                    self.js_parser(None, striped_content, pkg_id, we_train_batch_name)
                elif self.is_c_bash and root_node.type in ["string_content", "string", "raw_string"]:
                    import string
                    striped_content = content.strip(string.punctuation)
                    self.ex_sh_parser(pkg_id, striped_content, None, we_train_batch_name, is_original)

                url_parse_ret = url_parse.urlparse(content)
                if root_node.type == "word":
                    ret_appdex = " " + content
                # if content.startswith("http://") or content.startswith("https://") or content.startswith("ftp://"):
                #     ret_appdex = " URI"
                if url_parse_ret.scheme != "" and url_parse_ret.netloc != "":
                    ret_appdex = " URI"
                elif content.endswith((".com", ".org", ".net", ".cn", ".us", ".jp", ".uk")):
                    ret_appdex = " URI"
                if content in self.special_list:
                    ret_appdex = " ;"
                if content in self.ignore_list:
                    ret_appdex = ""
                if parents_type == "command_name":
                    ret_appdex = " " + command_name
                if content.endswith((".js", ".cjs", ".mjs")):
                    ret_appdex = " js_filepath"
                elif content.endswith(".sh"):
                    ret_appdex = " sh_filepath"
                self.bash_lv -= 1
                return ret + ret_appdex

    def js_parser(self, path, code, pkg_id, we_train_batch_name=None):
        if path:
            try:
                with open(path, "r") as f:
                    code = f.read()
                if len(code) > 30000:
                    is_large_file = True
                else:
                    is_large_file = False

                root_node = self.parse(code, "javascript")
            except:
                print("fail to open: " + path)
                return
        elif code:
            if len(code) > 30000:
                is_large_file = True
            else:
                is_large_file = False
            root_node = self.parse(code, "javascript")
        else:
            print("no js code input")
            return

        code_tokens = self.traverse_js_ast(root_node, is_large_file, pkg_id, we_train_batch_name)
        code_tokens = remove_api(code_tokens)

        for func_name, code in self.func_def_dict.items():
            if not self.func_use_dict[func_name] and is_large_file:
                code = remove_api(code)
                if code.strip() != "":
                    self.js_seq_dict[self.js_pkg_id] = " ".join(code.split(" ")).strip()
                    self.pkg_id_2_pkg_name_dict[self.js_pkg_id] = self.pkg_id_2_pkg_name_dict[pkg_id]
                    self.js_pkg_id -= 1
            elif not self.func_use_dict[func_name] and not is_large_file:
                code_tokens_list = code_tokens.split()
                for i in code_tokens_list:
                    if i == func_name:
                        code_tokens_list[code_tokens_list.index(i)] = code
                code_tokens = " ".join(code_tokens_list)

        self.func_def_dict = {}
        self.func_use_dict = {}
        if code_tokens.strip() != "":
            self.js_seq_dict[self.js_pkg_id] = " ".join(code_tokens.split(" ")).strip()
            self.pkg_id_2_pkg_name_dict[self.js_pkg_id] = self.pkg_id_2_pkg_name_dict[pkg_id]
            self.js_pkg_id -= 1

    def ex_sh_parser(self, pkg_id, code=None, path=None, we_train_batch_name=None, is_original=False):
        try:
            if code is None:
                with open(path, "r") as f:
                    code = f.read()
        except:
            print("fail to open: " + path)
            return
        try:
            root_node = self.parse(code, "bash")
        except:
            print("fail to parse:" + code)
            return
        self.cmd_pkg_ex_id -= 1
        self.pkg_id_2_pkg_name_dict[self.cmd_pkg_ex_id] = self.pkg_id_2_pkg_name_dict[pkg_id]
        print("ex_id: ", self.pkg_id_2_pkg_name_dict[pkg_id], self.cmd_pkg_ex_id)
        seq = self.traverse_bash_ast(root_node, self.cmd_pkg_ex_id, we_train_batch_name, is_original=is_original)
        seq = " ".join(seq.split())
        if seq != "":
            self.cmd_seq_dict[self.cmd_pkg_ex_id] = seq.strip()

    def cmd_canonical_sort(self, cmds_seq):
        def cosine_similarity(vec1, vec2):
            # 计算两个向量的内积
            dot_product = np.dot(vec1, vec2)
            # 计算两个向量的范数
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            # 计算余弦相似度
            cosine_sim = dot_product / (norm_vec1 * norm_vec2)
            return cosine_sim
        # print(self.bb_list)
        cmd_seq_list = cmds_seq.split(" ;")
        cmd_result_list = []
        id2cmd_dict = {}
        count = 0
        for cmd_seq in cmd_seq_list:
            dist_dict = {}
            cmd_name = cmd_seq.strip().split(" -")[0]
            # print("cmd_name:", cmd_name)
            # print(cmd_seq.strip().split(" -")[1:])
            for bb in cmd_seq.strip().split(" -")[1:]:
                count += 1
                # print("bb:", bb)
                try:
                    first_word = "-" + bb.strip().split()[0]
                except:
                    first_word = "-"
                # print("first word:", first_word)
                vec = self.cmd_wv_model.wv[first_word]
                id2cmd_dict[count] = bb.strip()
                dist_dict[count] = cosine_similarity(vec, self.cmd_centroid_vec)

            id_list = sorted(dist_dict, key=dist_dict.get)
            bb_list = []
            for id in id_list:
                bb_list.append(id2cmd_dict[id])
            if len(bb_list):
                cmd_seq = cmd_name + " -" + " -".join(bb_list)
            else:
                cmd_seq = cmd_name
            cmd_result_list.append(cmd_seq)

        return " " + " ; ".join(cmd_result_list)


def get_file_list(file_list, suffix, target_dir):
    for i in os.listdir(target_dir):
        if os.path.isfile(os.path.join(target_dir, i)) and i.endswith(suffix):
            file_list.append(os.path.join(target_dir, i))
        elif os.path.isdir(os.path.join(target_dir, i)):
            file_list = get_file_list(file_list, suffix, os.path.join(target_dir, i))
    return file_list


def get_random_from_json(benign_batch_name, benign_limit):
    def select_random_elements_from_json_file(file_path, benign_limit, new_file_path):
        with open(file_path, 'r') as file:
            # 从文件中加载 JSON 数据并解析为字典
            original_dict = json.load(file)

        keys = list(original_dict.keys())
        selected_keys = random.sample(keys, min(benign_limit, len(keys)))
        selected_dict = {key: original_dict[key] for key in selected_keys}
        sorted_dict = dict(sorted(selected_dict.items(), key=lambda x: float(x[0])))
        for k, v in sorted_dict.items():
            sorted_dict[k] = v.replace("\n", " ")

        with open(new_file_path, "w+") as fp:
            json.dump(sorted_dict, fp, indent=4)

    file_path_js_seq = os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", benign_batch_name, "raw_js_seq_all.json")
    file_path_cmd_seq = os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", benign_batch_name, "raw_cmd_seq_all.json")

    file_path_random_picked_js_seq = os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", benign_batch_name, "raw_js_seq.json")
    file_path_random_picked_cmd_seq = os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", benign_batch_name, "raw_cmd_seq.json")

    select_random_elements_from_json_file(file_path_js_seq, benign_limit, file_path_random_picked_js_seq)
    select_random_elements_from_json_file(file_path_cmd_seq, benign_limit, file_path_random_picked_cmd_seq)


def check_pkg_metadata(mode, target_dir, result_dir=None):
    def get_time_diff(target_dir, pkg):
        def run_with_timeout(timeout, target_dir, pkg):
            q = multiprocessing.Queue()
            p = multiprocessing.Process(args=(q,), target=lambda q: q.put(get_time(target_dir, pkg)))
            p.start()
            p.join(timeout)

            if p.is_alive():
                p.terminate()
                p.join()
                raise TimeoutError("Function call timed out.")
            return q.get()

        def get_time(target_dir, pkg):
            file_dir = os.path.join(target_dir, pkg, "package/package.json")
            try:
                with open(file_dir, "r") as f:
                    json_data = json.load(f)
                    version = json_data["version"]
                    pkg_name = json_data["name"]
                    pkg_full_name = json_data["name"] + "-" + version
                url = "https://registry.npmjs.org/" + pkg_name
                web_json_data = requests.get(url).content.decode("utf-8")
                web_json = json.loads(web_json_data)
                # print(web_json)
                iso_time = web_json["time"]["created"].strip("Z")
                given_time = datetime.datetime.fromisoformat(iso_time)
                # 计算时间差
                time_diff = current_time - given_time
                print(time_diff)
                # 判断时间差是否超过 90 天
                if time_diff.days < 90:
                    # tmp_cmd = "rm -r " + os.path.join(target_dir, pkg)
                    # os.system(tmp_cmd)
                    return pkg
            except:
                print("fail to get time:")
            return None

        try:
            pkg = run_with_timeout(120, target_dir, pkg)  # 设置最大执行时间为10秒
            return pkg
        except TimeoutError as e:
            print(e)
            return None

    # 准备pkg_list
    if mode == "train" or mode == "malicious" or mode == "batch":
        pkg_list = os.listdir(target_dir)
    elif mode == "benign":
        if not os.path.exists(os.path.join(result_dir, "checked_pkg_list.json")):
            pkg_list = os.listdir(target_dir)
            # 获取当前时间
            current_time = datetime.datetime.now()
            print(current_time)

            pool = ThreadPoolExecutor(max_workers=16)
            task_list = []
            for pkg in pkg_list:
                res = pool.submit(lambda cxp:get_time_diff(*cxp), (target_dir, pkg))
                task_list.append(res)

            for future in as_completed(task_list):
                result = future.result()
                if result:
                    if result in pkg_list:
                        pkg_list.remove(result)
            with open(os.path.join(result_dir, "checked_pkg_list.json"), 'w+') as f:
                json.dump(pkg_list, f, indent=4)
        else:
            with open(os.path.join(result_dir, "checked_pkg_list.json"), 'r') as f:
                return json.load(f)
    else:
        print("Warning: get_raw_samples_main: wrong input mode.")
        return None
    return pkg_list


def remove_dup(js_seq_dict, cmd_seq_dict):
    def process_seq_dict(seq_dict, dup_dict):
        tmp_dict = {}
        no_dup_dict = {}
        for pkg_id, seq in seq_dict.items():
            if seq in tmp_dict.keys():
                if tmp_dict[seq] not in dup_dict.keys():
                    dup_dict[tmp_dict[seq]] = [pkg_id]
                else:
                    dup_dict[tmp_dict[seq]].append(pkg_id)
            else:
                tmp_dict[seq] = pkg_id
                no_dup_dict[pkg_id] = seq

        return no_dup_dict, dup_dict

    js_seq_dict, js_dup_dict = process_seq_dict(js_seq_dict, {})
    cmd_seq_dict, cmd_dup_dict = process_seq_dict(cmd_seq_dict, {})

    js_seq_dict = dict(sorted(js_seq_dict.items(), key=lambda x: x[0]))
    cmd_seq_dict = dict(sorted(cmd_seq_dict.items(), key=lambda x: x[0]))

    return js_seq_dict, cmd_seq_dict, js_dup_dict, cmd_dup_dict


def get_raw_samples_main(target_dir, output_file_name_cmd_seq, output_file_name_js_seq, result_dir, mode,
                         we_train_batch_name, is_canonical, is_original):
    pkg_list = check_pkg_metadata(mode, target_dir, result_dir)
    # pkg_list = os.listdir(target_dir)
    if is_canonical:
        print("start loading model")
        js_wv_model, cmd_wv_model = load_model(we_train_batch_name)
        print("model loaded")

    target_script_dict = {}
    pkg_name_2_file_path_dict = {}
    pkg_id_2_pkg_name_dict = {}
    count = 0

    for file in pkg_list:
        file_dir = os.path.join(target_dir, file, "package/package.json")
        try:
            with open(file_dir, "r") as f:
                json_data = json.load(f)
                version = json_data["version"]
                pkg_name = json_data["name"] + "-" + version
                if "scripts" in json_data:
                    scripts = json_data["scripts"]
                    pkg_name_2_file_path_dict[pkg_name] = os.path.join(target_dir, file)
                    if mode != "malicious" and mode != "benign" and mode != "batch":
                        for name, script in json_data["scripts"].items():
                            count += 1
                            target_script_dict[count] = scripts[name]
                            pkg_id_2_pkg_name_dict[count] = pkg_name
                    else:
                        for name, script in json_data["scripts"].items():
                            if name == "postinstall" or name == "preinstall" or name == "install":
                                count += 1
                                target_script_dict[count] = scripts[name]
                                pkg_id_2_pkg_name_dict[count] = pkg_name
        except:
            print("fail to open: " + file_dir + "\n")

    if is_original:
        # cmd_seq_dict = target_script_dict
        # js_seq_dict = {}
        # print(target_script_dict)
        for k,v in target_script_dict.items():
            if type(v) == type("a"):
                target_script_dict[k] = v.replace("\n", "").replace("\r", "")
        my_parser = TrainParser(is_canonical, {}, {}, pkg_name_2_file_path_dict, "test", pkg_id_2_pkg_name_dict)
        for key, value in target_script_dict.items():
            print("success: ", pkg_id_2_pkg_name_dict[key])
            root_node = my_parser.parse(value, 'bash')
            if root_node:
                my_parser.traverse_bash_ast(root_node, key, is_original=is_original)
    else:
        if is_canonical:
            my_parser = TrainParser(is_canonical, {}, {}, pkg_name_2_file_path_dict, "test", pkg_id_2_pkg_name_dict)
            my_parser.js_wv_model = js_wv_model
            my_parser.cmd_wv_model = cmd_wv_model
            for key, value in target_script_dict.items():
                print("success: ", pkg_id_2_pkg_name_dict[key])
                root_node = my_parser.parse(value, 'bash')
                if root_node:
                    my_parser.traverse_bash_ast(root_node, key, we_train_batch_name)
        else:
            my_parser = TrainParser(is_canonical, {}, {}, pkg_name_2_file_path_dict, "test", pkg_id_2_pkg_name_dict)
            for key, value in target_script_dict.items():
                print("success: ", pkg_id_2_pkg_name_dict[key])
                root_node = my_parser.parse(value, 'bash')
                if root_node:
                    my_parser.traverse_bash_ast(root_node, key)

    if mode == "malicious" or mode == "batch":
        js_seq_dict, cmd_seq_dict, js_dup_dict, cmd_dup_dict = remove_dup(my_parser.js_seq_dict, my_parser.cmd_seq_dict)
        print("####################### js dup #######################")
        for k,v in js_dup_dict.items():
            print(k)
            print(v)
        print("####################### cmd dup #######################")
        for k,v in cmd_dup_dict.items():
            print(k)
            print(v)
    else:
        js_seq_dict = my_parser.js_seq_dict
        cmd_seq_dict = my_parser.cmd_seq_dict

    with open(output_file_name_cmd_seq, 'w') as json_file:
        json_file.write(json.dumps(cmd_seq_dict, indent=4))
    with open(output_file_name_js_seq, 'w') as json_file:
        json_file.write(json.dumps(js_seq_dict, indent=4))

    file_name_id2pkg = os.path.join(result_dir, "id2pkg.json")
    with open(file_name_id2pkg, "w+") as f:
        json_str = json.dumps(pkg_id_2_pkg_name_dict, indent=4)
        f.write(json_str)


def remove_api(seq):
    with open(os.path.join(CURRENT_DIR, "need_2_remove_apis.txt"), "r") as api_fp:
        target_api_list = api_fp.readlines()
    for target_api in target_api_list:
        seq_list = seq.split(" ")
        while target_api.strip() in seq_list:
            seq_list.remove(target_api.strip())
            seq = " ".join(seq_list)
    return seq


def load_model(we_train_batch_name):
    js_wv_model = FastText.load(
        os.path.join(CURRENT_DIR, "../DataShare/Model/WordEmbedding", we_train_batch_name, "js_we.model"))
    cmd_wv_model = FastText.load(
        os.path.join(CURRENT_DIR, "../DataShare/Model/WordEmbedding", we_train_batch_name, "cmd_we.model"))
    return js_wv_model, cmd_wv_model


def train_se_samples_parser_main(model_batch_name, target_dir, we_train_batch_name, is_canonical, is_original):
    if not os.path.exists(os.path.join(CURRENT_DIR, "../DataShare/Model")):
        os.mkdir(os.path.join(CURRENT_DIR, "../DataShare/Model"))
    if not os.path.exists(os.path.join(CURRENT_DIR, "../DataShare/Model/SeqEmbedding")):
        os.mkdir(os.path.join(CURRENT_DIR, "../DataShare/Model/SeqEmbedding"))
    if not os.path.exists(os.path.join(CURRENT_DIR, "../DataShare/Model/SeqEmbedding/" + model_batch_name)):
        os.mkdir(os.path.join(CURRENT_DIR, "../DataShare/Model/SeqEmbedding/" + model_batch_name))

    file_path_js_seq = os.path.join(CURRENT_DIR, "../DataShare/Model/SeqEmbedding/", model_batch_name, "js_train_samples.json")
    file_path_cmd_seq = os.path.join(CURRENT_DIR, "../DataShare/Model/SeqEmbedding/", model_batch_name, "cmd_train_samples.json")
    result_dir = os.path.join(CURRENT_DIR, "../DataShare/Model/SeqEmbedding/", model_batch_name)

    get_raw_samples_main(target_dir, file_path_cmd_seq, file_path_js_seq, result_dir, "train", we_train_batch_name, is_canonical, is_original)


def train_we_samples_parser_main(we_train_batch_name, target_dir, is_original):
    if not os.path.exists(os.path.join(CURRENT_DIR, "../DataShare/Model")):
        os.mkdir(os.path.join(CURRENT_DIR, "../DataShare/Model"))
    if not os.path.exists(os.path.join(CURRENT_DIR, "../DataShare/Model/WordEmbedding")):
        os.mkdir(os.path.join(CURRENT_DIR, "../DataShare/Model/WordEmbedding"))
    if not os.path.exists(os.path.join(CURRENT_DIR, "../DataShare/Model/WordEmbedding/" + we_train_batch_name)):
        os.mkdir(os.path.join(CURRENT_DIR, "../DataShare/Model/WordEmbedding/" + we_train_batch_name))

    file_path_js_seq = os.path.join(CURRENT_DIR, "../DataShare/Model/WordEmbedding/", we_train_batch_name, "js_train_samples.json")
    file_path_cmd_seq = os.path.join(CURRENT_DIR, "../DataShare/Model/WordEmbedding/", we_train_batch_name, "cmd_train_samples.json")
    result_dir = os.path.join(CURRENT_DIR, "../DataShare/Model/WordEmbedding/", we_train_batch_name)

    get_raw_samples_main(target_dir, file_path_cmd_seq, file_path_js_seq, result_dir, "train", we_train_batch_name, False, is_original)


def benign_samples_parser_main(benign_batch_name, target_dir, benign_limit, we_train_batch_name, is_canonical, is_original):
    if not os.path.exists(os.path.join(CURRENT_DIR, "../DataShare/BaseData")):
        os.mkdir(os.path.join(CURRENT_DIR, "../DataShare/BaseData"))
    if not os.path.exists(os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign")):
        os.mkdir(os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign"))
    if not os.path.exists(os.path.join(os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", benign_batch_name))):
        os.mkdir(os.path.join(os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", benign_batch_name)))

    output_dir = os.path.join(CURRENT_DIR, "../DataShare/BaseData/Benign", benign_batch_name)
    file_path_js_seq = os.path.join(output_dir, "raw_js_seq_all.json")
    file_path_cmd_seq = os.path.join(output_dir,  "raw_cmd_seq_all.json")

    # check_pkg_metadata("benign", target_dir, output_dir)

    get_raw_samples_main(target_dir, file_path_cmd_seq, file_path_js_seq, output_dir, "benign", we_train_batch_name, is_canonical, is_original)

    get_random_from_json(benign_batch_name, benign_limit)


def malicious_samples_parser_main(malicious_batch_name, target_dir, we_train_batch_name, is_canonical, is_original):
    if not os.path.exists(os.path.join(CURRENT_DIR, "../DataShare/BaseData")):
        os.mkdir(os.path.join(CURRENT_DIR, "../DataShare/BaseData"))
    if not os.path.exists(os.path.join(CURRENT_DIR, "../DataShare/BaseData/Malicious")):
        os.mkdir(os.path.join(CURRENT_DIR, "../DataShare/BaseData/Malicious"))
    if not os.path.exists(os.path.join(os.path.join(CURRENT_DIR, "../DataShare/BaseData/Malicious", malicious_batch_name))):
        os.mkdir(os.path.join(os.path.join(CURRENT_DIR, "../DataShare/BaseData/Malicious", malicious_batch_name)))

    output_dir = os.path.join(CURRENT_DIR, "../DataShare/BaseData/Malicious", malicious_batch_name)
    file_path_js_seq = os.path.join(output_dir, "raw_js_seq.json")
    file_path_cmd_seq = os.path.join(output_dir,  "raw_cmd_seq.json")

    get_raw_samples_main(target_dir, file_path_cmd_seq, file_path_js_seq, output_dir, "malicious", we_train_batch_name, is_canonical, is_original)


def batch_samples_parser_main(batch_name, target_dir, we_train_batch_name, is_canonical, is_original):
    result_dir = os.path.join(CURRENT_DIR, "../DataShare/Result")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if not os.path.exists(os.path.join(result_dir, batch_name)):
        os.mkdir(os.path.join(result_dir, batch_name))

    output_dir = os.path.join(CURRENT_DIR, "../DataShare/Result/", batch_name)
    file_path_js_seq = os.path.join(output_dir, "raw_js_seq.json")
    file_path_cmd_seq = os.path.join(output_dir,  "raw_cmd_seq.json")

    get_raw_samples_main(target_dir, file_path_cmd_seq, file_path_js_seq, output_dir, "batch", we_train_batch_name, is_canonical, is_original)


def merge_malicious_samples_main(malicious_batch_name):
    if not os.path.exists(os.path.join(os.path.join(CURRENT_DIR, "../DataShare/BaseData/Malicious", malicious_batch_name))):
        os.mkdir(os.path.join(os.path.join(CURRENT_DIR, "../DataShare/BaseData/Malicious", malicious_batch_name)))
    import shutil
    shutil.copy(os.path.join(CURRENT_DIR, "../DataShare/BaseData/Malicious/", malicious_batch_name + "-js", "raw_js_seq.json"), os.path.join(CURRENT_DIR, "../DataShare/BaseData/Malicious/", malicious_batch_name, "raw_js_seq.json"))
    shutil.copy(os.path.join(CURRENT_DIR, "../DataShare/BaseData/Malicious/", malicious_batch_name + "-sh", "raw_cmd_seq.json"), os.path.join(CURRENT_DIR, "../DataShare/BaseData/Malicious/", malicious_batch_name, "raw_cmd_seq.json"))


if __name__ == "__main__":
    model_batch_name = "20240219-1"
    benign_batch_name = "20240224-1"


