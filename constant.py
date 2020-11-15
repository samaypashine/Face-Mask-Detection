import os

"""
System Constants for Application
"""

"""
STATE: bool 
       True => DEBUG Mode
       False => DEPLOYMENT Mode
"""
STATE = False
"""
JETSON: bool
        True => working on Jetson Nano
        False => working on other Platform then Jetson Nano
"""
JETSON = False


"""
Networking Constants
"""


"""
TIMEOFWIFIRESTART: Int
               Sleeping Timing in Seconds after Restarting WIFI
"""
TIMEOFWIFIRESTART = 5


"""
HOTSPOTNAME:String
            SSID name or Access Point Name for creating Hotspot Name
"""
HOTSPOTNAME = "KiaraHotspot"


"""
HOTSPOTPASSWORD:String
                SSID password or Access Point password for creating Hotspot Name
"""
HOTSPOTPASSWORD = "kiara@123"


"""
WIFIINTERFACE :String
               Name of WIFI card Interface...
"""
WIFIINTERFACE = [interface for interface in os.listdir(
    '/sys/class/net/') if interface[0].lower() == 'w'][0]


"""
Jetson Nano Pin Constant
"""


"""
YELLOWPINNUMBER: Int
                 GPIO pin number to which yellow LED connected
                 This is according to GPIO.BOARD  
"""
YELLOWPINNUMBER = 33

"""
YELLOWBLINKINGTIME: Int
                    blinking time for Yellow LED which is sleeping time in after turn off and turn on
"""
YELLOWBLINKINGTIME = 1


"""
REDPINNUMBER: Int
              GPIO pin number to which red LED connected
              This is according to GPIO.BOARD
"""
REDPINNUMBER = 29


"""
REDBLINKINGTIME: Int
                blinking time for Red LED which is sleeping time in after turn off and turn on
"""
REDBLINKINGTIME = 1


"""
BUTTONPINNUMBER: Int
              GPIO pin number to which press button connected
              This is according to GPIO.BOARD
"""
BUTTONPINNUMBER = 13

"""
REDINDICATORPINNUMBER: Int
              GPIO pin number to which RED indicator LED connected
              This is according to GPIO.BOARD
"""
REDINDICATORPINNUMBER = 15


"""
GREENINDICATORPINNUMBER: Int
              GPIO pin number to which GREEN indicator LED connected
              This is according to GPIO.BOARD
"""
GREENINDICATORPINNUMBER = 11


"""
BUZZERPINNUMBER: Int
              GPIO pin number to which BUZZER connected
              This is according to GPIO.BOARD
"""
BUZZERPINNUMBER = 19


"""
System constants
"""


"""
TIMEOUTFORSHUTDOWN: Int
                    Timeout for Shutdown Button in milliseconds
                    i.e. button press and release below this time the system will shutdown
"""
TIMEOUTFORSHUTDOWN = 1000


"""
WAITINGTIMEFORWEBSTREAMING: Int
                           After disconnection of client, for how much time in seconds the server 
                           must stream the video or access the camera
"""
WAITINGTIMEFORWEBSTREAMING = 5

"""
SLEEPBEFOREREBOOT:Int
                    Time in seconds before reboot the systems
"""
SLEEPBEFOREREBOOT = 10


"""
File Path Constants
"""


"""
BASEPATH: string
          base path for package
"""
BASEPATH = os.path.dirname(os.path.realpath(__file__))

"""
PATHTOAPPFILEFORFLASK:String
                      absolute path to app.py file for flask application which contain the configurational mode
"""
PATHTOAPPFILEFORFLASK = BASEPATH+"/app.py"


"""
ACCESSCONFIGPATH:String
                 absolute path to access.config file for binary file read and write
"""
ACCESSCONFIGPATH = BASEPATH+"/access.config"


"""
MODELFILELOCATION:string
                  file location for inference model
"""
MODELFILELOCATION = BASEPATH + "/model"


"""
MODELSAVENAME: string
           model file name for saving model file
"""
MODELSAVENAME = "model.h5"


"""
CLASSTEXTSAVENAME: string
                   classification text file name for saving text file
"""
CLASSTEXTSAVENAME = "class.txt"

