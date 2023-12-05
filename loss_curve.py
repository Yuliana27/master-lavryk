import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def loss_curve(train_loss, val_loss, test_loss):
  loss_values = pd.DataFrame({"Train": train_loss,
                           "Val": val_loss,
                           "Test": test_loss})
  loss_values = loss_values.reset_index(drop=False).melt(id_vars ="index")
  loss_values = loss_values.rename(columns={"index":"Epoch", "value" : "RMSE", "variable" :"Sample"})

  fig = plt.figure(figsize=(10, 8))
  sns.set(style='whitegrid')
  sns.lineplot(x="Epoch", y="RMSE", hue="Sample", data=loss_values)
  plt.title("Loss Curve (RMSE)", fontsize=12)
  plt.show()