# coding=utf-8
# logging模块打印log日志到文件和屏幕
# Log级别：CRITICAL(50)、ERROR(40)、WARNING(30)、INFO(20)、DEBUG(10)、NOTSET(0)
# https://www.cnblogs.com/liujiacai/p/7804848.html
import logging,os,sys,time
 
class Logger(object):
    def __init__(self):
        # 指定日志文件名，获取当前执行的py文件名
        # 获取当天日期
        file_name_date = time.strftime("%Y-%m-%d", time.localtime())
        self.filename = str(os.path.basename(sys.argv[0]).split(".")[0]) + file_name_date + '.log'
        # 指定输出的格式和内容
        self.format = '%(asctime)s [%(filename)s] %(levelname)s:%(message)s'
        # 设置日志级别，默认为logging.WARNNING
        # self.level = logging.INFO
        # 和file函数意义相同，指定日志文件的打开模式，'w'或者'a'
        # self.filemode = 'a'
        # 指定时间格式
        self.datefmt = '%Y-%m-%d %H:%M:%S'
        self.logger = logging.getLogger(__name__)  # 创建日志记录器
 
    def new_file_logger(self,name=0):
        file_name_date = time.strftime("%Y-%m-%d", time.localtime())
        for i in range(10):
            default_name = str(os.path.basename(sys.argv[0]).split(".")[0]) + file_name_date + "_" + str(name) +"_"+str(i)+'.log'  
            if os.path.exists(default_name):
                continue
            self.filename = default_name
            break
        return self.get_logger()

    def get_logger(self):
        # 这个方法只能保存到文件，不能同时输出到屏幕
        # logging.basicConfig(filename=self.filename,format=self.format,level=self.level
        #                     ,filemode=self.filemode,datefmt=self.datefmt)
        self.logger.setLevel(logging.INFO)  # 日志记录器的等级
        format = logging.Formatter(self.format,datefmt=self.datefmt)
        # 日志输出到文件
        file_handler = logging.FileHandler(self.filename)  # 创建文件处理器
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(format)
        # 使用StreamHandler输出到屏幕
        console = logging.StreamHandler()  # 创建日志处理器
        console.setLevel(logging.INFO)
        console.setFormatter(format)
        # 添加两个Handler
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console)
        # 执行输出信息: 2019-01-02 15:51:13 [print_debug.py] INFO:输出信息内容
        # self.logger.info("输出信息内容")
        # 注意不能这么写,类型不一致
        # self.logger.info("经营许可证字段值个数:", 2)
        # 这么写就可以
        # self.logger.info("经营许可证字段值个数:%s"%2)
        # self.logger.info("经营许可证字段值个数:%s",3)
        # try:
        #     (a== 2)
        # except Exception as e:
        #     self.logger.info(e)
        # 如果这样返回，那样调取的时候就不用每次写logger.info，直接写logger就可以了
        return self.logger.info
if __name__ =="__main__":
    log_info = Logger()
    logger = log_info.get_logger()