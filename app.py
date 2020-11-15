from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    Response
)
import network
import constant
import os
from importlib import import_module
#from threadedLED import LED
from multiprocessing import Process
import time
import binaryFile
from werkzeug.utils import secure_filename
# import camera driver


if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera_opencv_face import Camera
if constant.JETSON:
    import Jetson.GPIO as GPIO

"""
Creating Flask Object
"""
app = Flask(__name__)

"""
Allowed Extensions for uploading files
"""
app.config['ALLOWED_EXTENSIONS'] = set(
    ['txt', 'h5'])

basedir = os.path.abspath(os.path.dirname(__file__))


def frameGen(camera):
    """
    Video streaming generator function.
    used to send frame header one by one using yield
    """
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


"""
ConfigureMode used to Handle state of website in flask and network connection
"""


class ConfigureMode():
    """
    Constructor creates the Hotspot and read access points and save that access point
    also initialize the front Message for user purpose.
    """

    def __init__(self):
        """
        Initialize network Objects and variables such as accessPoints, MessageForFront.
        """
        '''
        if constant.JETSON:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setwarnings(False)
        else:
            print("Board set as GPIO.BOARD")
        
        LED("buzzer", "stableOff").start()
        LED("red", "stableOff").start()
        LED("ired", "stableOff").start()
        LED("igreen", "stableOff").start()
        LED("yellow", "stableOff").start()
        '''
        #self.connection = network.Network()
        #self.connection.turnOffWifi()
        #self.connection.turnOnWifi(constant.TIMEOFWIFIRESTART)
        #self.connection.createHotspot(
        #    constant.HOTSPOTNAME, constant.HOTSPOTPASSWORD, constant.WIFIINTERFACE)
        
        #if(constant.STATE):
        #    print("Current IP address:", self.connection.getIPAddress())

    def extensionCheck(self, filename):
        """
        Check if File Extension is valid or not
        return type:bool
                    True if extension is valid
                    Flase if extension is invalid
        """
        return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

    def h5ExtensionCheck(self, filename):
        """
        Check if File Extension is .h5 or not
        return type:bool
                    True if extension is .h5
                    Flase if extension is otherwise
        """
        print(filename.rsplit('.', 1)[1] == 'h5')
        return '.' in filename and filename.rsplit('.', 1)[1] == 'h5'

    def txtExtensionCheck(self, filename):
        """
        Check if File Extension is .txt or not
        return type:bool
                    True if extension is .txt
                    Flase if extension is otherwise
        """

        return '.' in filename and filename.rsplit('.', 1)[1] == 'txt'


mode = ConfigureMode()


@app.route('/')
def index():
    """
    request url: / or root
    render page: index.html
    variables:
        message type:string
                    :use to show message according to action of user's
    """
    return render_template('index.html')


@app.route('/configure')
def configure():
    """
    request url: /configure
    render page: configure.html
    variables:
        message type:string
                    :use to show message according to action of user's
    """
    return render_template('configure.html')


@app.route('/upload', methods=['POST'])
def upldfile():
    """
    request url: /upload
    method : POST
    Actions : Store the file uploaded through call
    return :string
            the message according to output of task
    """
    if request.method == 'POST':
        try:
            model = request.files['model']
            modelname = secure_filename(model.filename)

            classes = request.files['class']
            classesname = secure_filename(classes.filename)

            modelFlag = False
            classFlag = False

            if model and mode.h5ExtensionCheck(model.filename):
                modelFlag = True
                print("Its True")
            else:
                if '.' in modelname:
                    return jsonify(message=modelname.rsplit('.', 1)[1]+" Extension Not Allowed For Model File:", status="415")
                else:
                    return jsonify(message="Model File must have Extension", status="415")
            if classes and mode.txtExtensionCheck(classes.filename):
                classFlag = True
            else:
                if '.' in classesname:
                    return jsonify(message=classesname.rsplit('.', 1)[1]+" Extension Not Allowed For Class File:", status="415")
                else:
                    return jsonify(message="Class File must have Extension", status="415")

            if(modelFlag and classFlag):
                modelname = secure_filename(model.filename)
                classname = secure_filename(classes.filename)
                updir = os.path.join(basedir, constant.MODELFILELOCATION+'/')
                model.save(os.path.join(updir, constant.MODELSAVENAME))
                classes.save(os.path.join(updir, constant.CLASSTEXTSAVENAME))
                return jsonify(message=modelname+" and "+classname+" Uploaded Sucessfully:", status="202")
        except Exception as e:
            return jsonify(message="File Can't be Empty", status="415")
    else:
        return jsonify(message="Method not Allowed", status="405")


@app.route('/streaming')
def streaming():
    """
    request url: /streaming
    response: frame header + frame for the streaming of video
    """
    return Response(frameGen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reboot', methods=['POST'])
def reboot():
    """
    request url: /reboot
               : reboot system
    """
    #yellow = LED('yellow', 'blinking')
    #yellow.start()
    #frameGen(Camera())
    #yellow.killThread()
    return "Optimization Done"


if __name__ == '__main__':
    frameGen(Camera())
    app.run(host='0.0.0.0', threaded=True)
