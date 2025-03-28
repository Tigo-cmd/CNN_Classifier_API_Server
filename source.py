#!/usr/bin/env python3
########################################################################
#       AN OPEN/GROQ AI CLI INTEGRATION MY NWALI UGONNA EMMANUEL
#       GITHUB: https://github.com/Tigo-cmd/TigoAi
#       All contributions are welcome!!!
#       yea lets do some coding!!!!!!!!!!!!
#########################################################################
"""Source classes to handle all GroqAPI functionalities and trained models
                        BY Nwali Ugonna Emmanuel
"""

from __future__ import annotations
from groq import Groq
import subprocess
import os
import requests
from dotenv import load_dotenv


class TigoGroq:
    """this handles Api tokens and requests
    this class clones the GROQ init function and re-initializes the arguments
    so that when the class is called it set every functionality needed for the model to run
    """
    client = None
    _Apikey = os.getenv("GROQ_API_KEY")
    context: list[dict[str, str]] = [
        {"role": "system", "content": "You are an expert plant disease assistant specialized in diagnosing and providing solutions for tomato plant diseases and very concise in your response. "
        "Your goal is to help users identify and manage tomato plant health issues based on their queries."
        "When a user asks a question, provide clear and concise answers. If they ask about specific symptoms (e.g., yellowing leaves, black spots, wilting), explain the possible diseases, causes, and recommended treatments. If the user is unsure about the issue, guide them with follow-up questions to narrow down the problem."
        "Additionally, if a user mentions uploading an image, encourage them to use the upload feature and let them know you'll analyze it. Always provide practical advice, such as natural remedies, chemical treatments, and prevention tips.Maintain a friendly and professional tone, and keep responses simple and easy to understand for farmers and plant growers. If a user asks about other plants, gently redirect them to focus on tomatoes."}
    ]

    def __init__(
            self,
    ) -> None:
        """Constructor initialized at first call"""
        if self._Apikey is None:
            """
            checks if the apikey is present in the environment variable
             else loads from env file using python-loadenv
            """
            load_dotenv()
        self.client = Groq()

    # def get_context(self, context: str):
    #     """

    #     :param context: tracks conversations and context with users
    #     :return: nothing
    #     """
    #     pass

    async def get_response_from_ai(self, message: str):
        """returns response from the AI and messages to print to standard output"""
        self.context.append({"role": "user", "content": message})
        completion = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=self.context,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        reply: str = ""
        for chunk in completion:
            reply += chunk.choices[0].delta.content or ""
        self.context.append({"role": "assistant", "content": reply})
        return reply

    # def store_retrive_context(self, filename: str):
    #     """

    #     :param filename:
    #     :return:
    #     """
    #     pass
