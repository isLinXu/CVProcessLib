import json


class JsonOperator:
    """
     操作 json文件 的 类
    """

    @staticmethod
    def load_jsonfile(filename=""):
        if "" == filename:
            exception = Exception("Error: filename is empty")
            raise exception
        else:
            with open(filename, "r", encoding="utf-8") as fp:
                return json.load(fp=fp)

    @staticmethod
    def write_jsonfile(content=None, filename=""):
        if "" == filename:
            exception = Exception("Error: filename is empty")
            raise exception
        else:
            with open(filename, "w", encoding="utf-8") as fp:
                json.dump(content, fp=fp, ensure_ascii=False, indent=4)  # 参数indent: json文件按格式写入, 距行首空4格;


def get_newData_with_sumPosition(first_dic=None, second_dic=None):
    new_dic = {}
    for (key, value) in first_dic.items():
        value['pos'] = first_dic[key]['pos'] + second_dic[key]['pos']
        new_dic[key] = value
    return new_dic


def main_function():
    # json数据源文件
    cta_hla_filename = "cta_strategy_data_2503023_hla.json"
    cta_hsr_filename = "cta_strategy_data_2503023_hsr.json"

    # 写入目标文件
    cta_target_filename = "cta_strategy_data_2503023_sum.json"

    # 读取json文件数据
    cta_hla_dic = JsonOperator().load_jsonfile(filename=cta_hla_filename)
    cta_hsr_dic = JsonOperator().load_jsonfile(filename=cta_hsr_filename)

    # 获取新的要写入json文件的数据
    new_dic = get_newData_with_sumPosition(cta_hla_dic, cta_hsr_dic)

    # 写入目标文件夹
    JsonOperator().write_jsonfile(content=new_dic, filename=cta_target_filename)


if __name__ == '__main__':
    main_function()