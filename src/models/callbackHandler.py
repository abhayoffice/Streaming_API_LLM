from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler

class MyStreamingCallback(StreamingStdoutCallbackHandler):
 def __init__(self):
  self.content = ""
  self.final_answer = False

  async def on_lm_new_token(self, token: TokenizedLMOutput):
   self.content += token
   if "final_answer" in self.content:
    self.final_answer = True
    self.content = ""
   if self.final_answer:
    print(token)
   agent.callbacks = MyStreamingCallback()