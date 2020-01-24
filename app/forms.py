from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField, HiddenField, StringField
from wtforms.validators import DataRequired

class CanvasForm(FlaskForm):
    # DataRequired validator makes it so user must click a star
    x = IntegerField('x', validators=[DataRequired()])
    y = IntegerField('y', validators=[DataRequired()])
    '''I will implement this field shortly. It will allow the user to chose the
    name of the output folder''' 
    # outputFolder = StringField('outputFolder', validators=[DataRequired()])
    submit = SubmitField('Track')
