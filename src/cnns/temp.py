import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Dados de exemplo
np.random.seed(10)
data = pd.DataFrame({
    'grupo': np.repeat(['A', 'B', 'C'], 100),
    'valores': np.concatenate([np.random.normal(0, std, 100) for std in range(1, 4)])
})

# Criando o boxplot
sns.boxplot(x='grupo', y='valores', data=data)
plt.title('Exemplo de Gráfico de Boxplots')
plt.show()
