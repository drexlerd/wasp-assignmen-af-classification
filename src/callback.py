from smac.callback import Callback


class CustomCallback(Callback):

  def on_end(self, smbo):
    print(smbo)