# Source: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world

from flask import Flask
from config import Config

application = Flask(__name__)  # __name__ variable is predefined
application.config.from_object(Config)  # config is name of file, Config is name of Class

from app import routes  # Putting this import down here prevents circular import
