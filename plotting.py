# libraries and data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make a data frame
df = pd.read_csv('../csv_files/coma_scaling.csv')

# style
plt.style.use('seaborn-darkgrid')

# create a color palette
palette = plt.get_cmap('Set1')

# multiple line plot
num = 0
for column in ['Reward_2', 'Reward_3', 'Reward_4', 'Reward_5']:
    num += 1
    plt.plot(df['Epoch'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

# Add legend
plt.legend(loc=2, ncol=2)

# Add titles
plt.title("COMA Training Curves for Different Numbers of Agents", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Episodes (x30)")
plt.ylabel("Reward per Agent")

plt.show()
#if __name__ == "__main__":

    #plot_learning_curves('../csv_files/coma_scaling.csv')