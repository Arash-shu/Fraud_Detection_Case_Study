#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template
import psycopg2
import pickle


app = Flask(__name__)


@app.route('/hello')
def api_root():
   return 'Hello, World! Fraud stops HERE!! '


@app.route('/dashboard/')
def results_display():
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT object_id,fraud_probability,fraud_risk FROM predictions ORDER BY ID DESC LIMIT 10")
        data = cur.fetchall()
        
        return render_template("dashboard.html", data=data)

    except Exception as e:
        return (str(e))



if __name__ == '__main__':
    global loaded_model
    global conn

    
    #Unpickle file
    with open('finalized_model.sav', 'rb') as f:
        loaded_model = pickle.load(f)
        
    #connect to database
    conn_string = "host='localhost' dbname='casestudy' user='tom' password='jerry'"
    conn = psycopg2.connect(conn_string)
    
    app.run()
    conn.close()