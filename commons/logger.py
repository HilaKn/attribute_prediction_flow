import logging
import os

logs_folder = os.path.join(os.path.dirname(__file__),"../logs")
if not os.path.exists(logs_folder):
    os.makedirs(logs_folder)

logFormatter = logging.Formatter("%(asctime)s %(module)s [%(funcName)s] [%(levelname)s]  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler(os.path.join(logs_folder, "attribute_prediction_flow.log"))
fileHandler.setLevel(logging.DEBUG)
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)