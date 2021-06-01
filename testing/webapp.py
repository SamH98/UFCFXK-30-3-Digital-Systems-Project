from flask import Flask, render_template, url_for, request, redirect, make_response
import pandas as pd



app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def main():
    alert = "No new devices found"
    hosts = pd.read_excel('devices.xlsx')
    return render_template('devices.html', tables=[hosts.to_html(classes='hosts', index=False)],
                           titles=['na', ''], variable=alert)


@app.route('/detection', methods=["GET", "POST"])
def detection():
    alert = 'No detections'

    annomalies = pd.read_excel('annomalies.xlsx')
    return render_template('detection.html', tables=[annomalies.to_html(classes='hosts', index=False)],
                           titles=['na', ''], variable=alert)


@app.route('/graphs', methods=["GET", "POST"])
def graphs():
    toplot = pd.read_excel('graphdata.xlsx')

    line_labels = toplot['time']
    line_values = toplot['in']
    line_labels1 = toplot['out']
    return render_template('graphs.html', title='Total data in and out every cycle', max=1000000, labels=line_labels,
                           values=line_values, values1=line_labels1)




if __name__ == "__main__":
    app.run(debug=True)
