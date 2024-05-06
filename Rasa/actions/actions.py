from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests

class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        intent =  tracker.latest_message['intent'].get('name')
        message = tracker.latest_message['text']
        dispatcher.utter_message(text=intent)
        # print(intent)
        # if intent == "greets":
        #     reponse = requests.post("http://127.0.0.1:5001/chatchit", json={"user_question": message}) 
        #     dispatcher.utter_message(text=reponse.text)
            # dispatcher.utter_message(text=response.text)
        # elif intent == "question":
        #     response = requests.post("http://127.0.0.1:5001/chatbot", json={"user_question": message})

            # suggest = requests.post("http://127.0.0.1:5001/suggest", json={"user_question": message})
            # dispatcher.utter_message(text=response.text)
            # dispatcher.utter_message(text=suggest.text)
        # suggest = requests.post("http://127.0.0.1:5001/suggest", json={"user_question": message})
        # dispatcher.utter_message(text=suggest.text)
        return []
