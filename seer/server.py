import argparse
from flask import Flask, render_template, flash, request
from wtforms import DateTimeField, Form, IntegerField, TextField, TextAreaField, validators, SelectField, StringField, SubmitField
from sklearn.externals import joblib
import numpy as np

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

# SciKit-Learn related
scalar = joblib.load('scalar.pickle')
model = joblib.load('model_rf.pickle')
with open('feature_selected_column_info.txt', 'r') as f:
    column_info = f.read().split()


class ReusableForm(Form):
    mar_stat = SelectField(label='Marital status:', choices=[
        ('1', 'Single (never married)'),
        ('2', 'Married (including common law)'),
        ('3', 'Separated'),
        ('4', 'Divorced'),
        ('5', 'Widowed'),
        ('6', 'Unmarried or domestic partner (same sex or opposite sex or unregistered)')],
        validators=[validators.required()])
    race1v = SelectField(label='Race:', choices=[
        ('01', 'White'),
        ('02', 'Black'),
        ('03', 'American Indian, Aleutian, Alaskan Native or Eskimo (includes all indigenous populations of the Western hemisphere)'),
        ('04', 'Chinese'),
        ('05', 'Japanese'),
        ('06', 'Filipino'),
        ('07', 'Hawaiian'),
        ('08', 'Korean'),
        ('10', 'Vietnamese'),
        ('11', 'Laotian'),
        ('12', 'Hmong'),
        ('13', 'Kampuchean (including Khmer and Cambodian)'),
        ('14', 'Thai'),
        ('15', 'Asian Indian or Pakistani, NOS'),
        ('16', 'Asian Indian'),
        ('17', 'Pakistani'),
        ('20', 'Micronesian, NOS'),
        ('21', 'Chamorran'),
        ('22', 'Guamanian, NOS'),
        ('25', 'Polynesian, NOS'),
        ('26', 'Tahitian'),
        ('27', 'Samoan'),
        ('28', 'Tongan'),
        ('30', 'Melanesian, NOS'),
        ('31', 'Fiji Islander'),
        ('32', 'New Guinean'),
        ('96', 'Other Asian, including Asian, NOS and Oriental, NOS'),
        ('97', 'Pacific Islander, NOS'),
        ('98', 'Other')],
        validators=[validators.required()])
    nhiade = SelectField(label='Hispanic origin:', choices=[
        ('0', 'Non-Spanish-Hispanic-Latino'),
        ('1', 'Mexican'),
        ('2', 'Puerto Rican'),
        ('3', 'Cuban'),
        ('4', 'South or Central American excluding Brazil'),
        ('5', 'Other specified Spanish/Hispanic Origin including Europe'),
        ('6', 'Spanish/Hispanic/Latino, NOS'),
        ('7', 'NHIA Surname Match Only'),
        ('8', 'Dominican Republic')],
        validators=[validators.required()])
    sex = SelectField(label='Gender:', choices=[
        ('1', 'Male'),
        ('2', 'Female')],
        validators=[validators.required()])
    age_dx = IntegerField(label='Age:', validators=[validators.required()])
    age_1rec = SelectField(label='Age group:', choices=[
        ('00', 'Age 00'),
        ('01', 'Ages 01-04'),
        ('02', 'Ages 05-09'),
        ('03', 'Ages 10-14'),
        ('04', 'Ages 15-19'),
        ('05', 'Ages 20-24'),
        ('06', 'Ages 25-29'),
        ('07', 'Ages 30-34'),
        ('08', 'Ages 35-39'),
        ('09', 'Ages 40-44'),
        ('10', 'Ages 45-49'),
        ('11', 'Ages 50-54'),
        ('12', 'Ages 55-59'),
        ('13', 'Ages 60-64'),
        ('14', 'Ages 65-69'),
        ('15', 'Ages 70-74'),
        ('16', 'Ages 75-79'),
        ('17', 'Ages 80-84'),
        ('18', 'Ages 85+')],
        validators=[validators.required()])
    mdxrecmp = IntegerField(label='Diagnosis Month:',
                            validators=[validators.required()])
    year_dx = IntegerField(label='Diagnosis Year:',
                           validators=[validators.required()])
    seq_num = SelectField(label='Tumor sequence:', choices=[
        ('00', '0'),
        ('01', '1'),
        ('02', '2')],
        validators=[validators.required()])
    primsite = SelectField(label='Primary site - I:', choices=[
        ('C180', 'Cecum'),
        ('C182', 'Ascending colon'),
        ('C183', 'Hepatic flexure of colon'),
        ('C184', 'Transverse colon'),
        ('C185', 'Splenic flexure of colon'),
        ('C186', 'Descending colon'),
        ('C187', 'Sigmoid colon'),
        ('C199', 'Rectosigmoid junction'),
        ('C209', 'Rectum, NOS')],
        validators=[validators.required()])
    histo2v = SelectField(label='Tumor morphology:', choices=[
        ('8140', 'Adenoma, NOS'),
        ('8210', 'Adenomatous polyp, NOS'),
        ('8261', 'Villous adenoma, NOS'),
        ('8263', 'Tubulovillous adenoma, NOS'),
        ('8480', 'Mucinous adenoma')],
        validators=[validators.required()])
    histo3v = SelectField(label='Tumor histology:', choices=[
        ('8140', 'Adenocarcinoma in situ, NOS'),
        ('8210', 'Adenocarcinoma in adenomatous polyp'),
        ('8261', 'Adenocarcinoma in villous adenoma'),
        ('8263', 'Adenocarcinoma in tubolovillous adenoma'),
        ('8480', 'Mucinous adenocarcinoma')],
        validators=[validators.required()])
    icdot10v = SelectField(label='Primary site - II:', choices=[
        ('C180', 'Cecum'),
        ('C181', 'Appendix'),
        ('C182', 'Ascending colon'),
        ('C183', 'Hepatic flexure of colon'),
        ('C184', 'Transverse colon'),
        ('C185', 'Splenic flexure of colon'),
        ('C186', 'Descending colon'),
        ('C187', 'Sigmoid colon'),
        ('C199', 'Rectosigmoid junction'),
        ('C209', 'Rectum')],
        validators=[validators.required()])
    histrec = SelectField(label='Disease groups:', choices=[
        ('05', '8140-8389 : adenomas and adenocarcinomas'),
        ('08', '8440-8499 : cystic, mucinous and serous neoplams')],
        validators=[validators.required()])
    cs0204schema = SelectField(label='Particular schema:', choices=[
        ('018', 'Colon'),
        ('130', 'Rectum')],
        validators=[validators.required()])
    grade = SelectField(label='Tumor grade:', choices=[
        ('1', 'Grade I; well differentiated; differentiated, NOS'),
        ('2', 'Grade II; moderately differentiated; intermediate differentiation'),
        ('3', 'Grade III; poorly differentiated'),
        ('4', 'Grade IV; undifferentiated; anaplastic'),
        ('5', 'T-cell; T-precursor'),
        ('6', 'B-cell; Pre-B; B-Precursor'),
        ('7', 'Null cell; Non T-non B'),
        ('8', 'N K cell (natural killer cell)')],
        validators=[validators.required()])
    hst_stga = SelectField(label='Tumor stage:', choices=[
        ('0', 'In situ —A noninvasive neoplasm'),
        ('1', 'Localized — An invasive neoplasm confined entirely to the organ of origin'),
        ('2', 'Regional — A neoplasm that has extended'),
        ('4', 'Distant — A neoplasm that has spread to parts of the body')],
        validators=[validators.required()])
    rept_src = SelectField(label='Reporting Source:', choices=[
        ('1', 'Hospital inpatient;unified medical records'),
        ('2', 'Radiation Treatment Centers or Medical Oncology Centers'),
        ('3', 'Laboratory Only'),
        ('4', 'Physician’s Office/Private Medical Practitioner'),
        ('5', 'Nursing/Convalescent Home/Hospice'),
        ('6', 'Autopsy Only'),
        ('7', 'Other hospital outpatient units/surgery centers')],
        validators=[validators.required()])
    beho3v = SelectField(label='Malignancies:', choices=[
        # ('0', 'Benign (Reportable for intracranial and CNS sites only)'),
        ('1', 'Uncertain whether benign or malignant, borderline malignancy, low malignant potential, and uncertain malignant potential'),
        ('2', 'Carcinoma in situ; noninvasive'),
        ('3', 'Malignant, primary site; invasive')],
        validators=[validators.required()])
    lateral = SelectField(label='Side of tumor orgin:', choices=[
        ('0', 'Not a paired site'),
        ('1', 'Right: origin of primary'),
        ('2', 'Left: origin of primary'),
        ('3', 'Only one side involved, right or left origin unspecified'),
        ('4', 'Bilateral involvement, lateral origin unknown; stated to be single primary'),
        ('5', 'Paired site: midline tumor')],
        validators=[validators.required()])


@app.route('/', methods=['GET'])
def survey():
    form = ReusableForm(request.form)
    return render_template('main.html', form=form)


@app.route('/result', methods=['POST'])
def result():
    print(type(request.form))
    # Run prediction
    num_features = len(column_info)
    X = np.zeros((1, num_features), dtype=np.float32)

    def add_categorical(name):
        val = request.form[name]
        feat_name = name.upper() + '__' + str(val)
        if feat_name in column_info:
            feat_idx = column_info.index(feat_name)
            X[0, feat_idx] = 1.0
        else:
            print('Feature {} not found'.format(feat_name))

    def add_numeric(name):
        val = float(request.form[name])
        feat_name = name.upper()
        if feat_name in column_info:
            feat_idx = column_info.index(feat_name)
            X[0, feat_idx] = val
        else:
            print('Feature {} not found'.format(feat_name))

    add_categorical('mar_stat')
    add_categorical('race1v')
    add_categorical('nhiade')
    add_categorical('sex')
    add_categorical('beho3v')
    add_categorical('lateral')
    add_categorical('grade')
    add_categorical('primsite')
    add_categorical('seq_num')
    add_categorical('histo2v')
    add_categorical('histo3v')
    add_categorical('rept_src')
    add_categorical('age_1rec')
    add_categorical('icdot10v')
    add_categorical('histrec')
    add_categorical('cs0204schema')
    add_categorical('hst_stga')

    add_numeric('age_dx')
    add_numeric('mdxrecmp')
    add_numeric('year_dx')

    X_scaled = scalar.transform(X)
    y = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)

    flash('5-year survival probability after surgery:')
    flash(str(y_prob[0][1] * 100) + ' %')
    # else:
    #     return('Error: All the form fields are required. ')
    # flash('Error: All the form fields are required. ')

    return render_template('result.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='127.0.0.1')
    parser.add_argument("--port", type=int, default='5000')
    args = parser.parse_args()

    app.run(args.host, args.port)
