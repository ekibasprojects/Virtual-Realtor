#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request
import requests
from twilio.twiml.messaging_response import MessagingResponse
from bot import RealtorBot

app = Flask(__name__)

app_started = False

@app.route('/', methods=['POST'])
def home():
    global app_started
    global bot
    
    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()
    
    if incoming_msg == '0':
        app_started = True
        bot = RealtorBot()
    else:
        if not app_started:
            msg.body("Привет!\nЯ - виртуальный риэлтор. Я могу оценить вашу квартиру для продажи.\n\nВведи *0*, чтобы начать.")
            return str(resp)
        
    process_output = ""
    if bot.started:
        process_output = bot.process_input(incoming_msg)
        
    if bot.started and process_output != "OK" and process_output != "END":
        output = process_output
    elif process_output == "END":
        app_started = False
        output = bot.input_prompt()
    else:
        output = bot.input_prompt()

    msg.body(output)
    return str(resp)

        
if __name__ == '__main__':
    app.run(debug=True)
           
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        