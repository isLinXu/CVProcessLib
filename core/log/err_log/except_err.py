# -*- coding: utf-8 -*-
import sys
import logging
from logging.handlers import RotatingFileHandler
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
# 定义一个RotatingFileHandler，最多备份3个日志文件，每个日志文件最大1K
rHandler = RotatingFileHandler("log.txt", maxBytes=10 * 1024 * 1024, backupCount=30)
rHandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
rHandler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(rHandler)
logger.addHandler(console)


logger.info("Start print log")
logger.debug("Do something")
logger.warning("Something maybe fail.")
logger.info("Finish")

def err_log(name, e):
    err = "Method：", name, "Tips:", str(e)
    logger.error(err)


def pose_log(name, e):
    err = "ImageName:", name, "Locals:", str(e)
    logger.info(err)


def try_except(f, name, *params):
    err = ""
    try:
        f(*params)
    except Exception as e:
        err = "Method：", name, "Tips:", str(e)
        logger.error(err)
        # logger.error("Method：",name,"Tips:",e)
        # print("Method：",name,"Tips:",e)
    return err


def try_except_form(req, params, info_params, info="info", is_save=True):
    err = ""
    name = None
    for p in params:
        name = p
        try:
            if name == "file":
                req.files[p]
            else:
                req.form[p]
        except Exception as e:
            err = "Method：", name, "Tips:", str(e)
            if is_save == True:
                logger.error(err)
            break
    if name == None:
        err = "Method：", name, "Tips:name为空"
    elif len(info_params) > 0:
        infos = eval(req.form[info])
        for param in info_params:
            name = param
            try:
                infos[param]
            except Exception as e:
                err = "Method：", name, "Tips:", str(e)
                if is_save == True:
                    logger.error(err)
                break
        if name == None:
            err = "Method：", name, "Tips:name为空"
    return err


if __name__ == "__main__":
    pose_log("sss","dasdsa")
    def sss(a, b, c, aa, bb):
        dic = [1, 2]
        dic[3]
        print(a, b, c, aa, bb)
    # 1
    try_except(sss, "sss", 1, 2, 3, {"1": "2"}, [1])
    # 2
    try:
        sss("sss", 1, 2, 3, {"1": "2"}, [1])
    except Exception as e:
        err_log("sss", e)
