import os

class Config(object):
    # Note: If this application is ever put online, use a better secret key to prevent CSRF attacks
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'secret_key'
