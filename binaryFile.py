import constant
from cryptography.fernet import Fernet

"""
Use to for Reading and writing the Binary file
using Encryption.
"""


class BinaryFile:
    """
    Initializing fixed Path to read and write the
    in file.
    """

    def __init__(self):
        self.file_path = constant.ACCESSCONFIGPATH

    def writeAccess(self, SSIDName, SSIDPassword):
        """
        Write SSIDName and SSIDPassword inside the file encrypted 

        :type SSIDName: String
        :param SSIDName: String value for SSID name or Access point name

        :type SSIDPassword: String
        :param SSIDPassword: String value for password of SSID name or Access point password

        """
        try:
            with open(self.file_path, "wb") as f:
                key = Fernet.generate_key()
                fe = Fernet(key)
                SSIDName = fe.encrypt(SSIDName.encode("utf-8"))+b'\n'
                SSIDPassword = fe.encrypt(
                    SSIDPassword.encode("utf-8"))+b'\n'
                f.write(SSIDName)
                f.write(SSIDPassword)
                f.write(key)
        except IOError as e:
            print("Writing Access.config: Something Went Wrong:")
        if constant.STATE:
            print("Writing Operation Completed Successfully:")

    def readAccess(self):
        """
        Read SSIDName and SSIDPassword by decrypting files

        :return type: List
        :return param: List of data contain SSIDName and SSIDPassword
        """
        tempList = list()
        accessList = list()
        try:
            with open(self.file_path, "rb") as f:
                for line in f:
                    tempList.append(line)
            key = tempList[2]
            fe = Fernet(key)
            accessList.append(fe.decrypt(tempList[0]).decode("utf-8"))
            accessList.append(fe.decrypt(tempList[1]).decode("utf-8"))
            if constant.STATE:
                print("Current Key:", key)
        except IOError as e:
            print("Can't Open or Read File")
            return list()
        if constant.STATE:
            print("Reading Operation Completed Successfully:")
        return accessList
