# Importar bibliotecas necessárias
import pandas as pd
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

# Carregar o arquivo usando o método do professor para arquivos desorganizados
from google.colab import files

# Remove todos os arquivos temporários enviados anteriormente
!rm -rf /content/*

# Fazer upload do arquivo
uploaded = files.upload()

#-Fim da primeira célula-

# Lendo o arquivo SMSSpamCollection com tabulação como separador
df = pd.read_csv(io.BytesIO(uploaded['SMSSpamCollection']), sep='\t', header=None, names=['target', 'text'])

# Verificar a leitura correta
print(df.head())

#-Fim da segunda célula-

# Conversão da coluna 'target' para valores binários (0 para 'ham', 1 para 'spam')
df['target'] = df['target'].map({'ham': 0, 'spam': 1})

# Separar variáveis independentes e dependentes
X = df['text']  # Features (mensagens)
y = df['target']  # Target (spam ou ham)

# Convertendo o texto para uma matriz TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf.fit_transform(X)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Treinando o modelo de Regressão Logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Gerando as probabilidades preditas para a classe positiva (spam)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Definir os thresholds de probabilidade
thresholds = [0.3, 0.7]

# Definindo bins
bins = np.linspace(0, 1, 11)

# Calculando as contagens
counts_general, _ = np.histogram(y_pred_prob, bins=bins)
counts_spam, _ = np.histogram(y_pred_prob[y_test == 1], bins=bins)

# Convertendo as contagens para porcentagens
percentages_general = counts_general / counts_general.sum() * 100
percentages_spam = counts_spam / counts_spam.sum() * 100

# Configurando o gráfico
bar_width = 0.4  # Largura das barras

# Plotando os histogramas
bar_positions = np.arange(len(bins) - 1)  # Posições das barras
plt.bar(bar_positions - bar_width/2, percentages_general, width=bar_width, color='blue', label='População Geral', alpha=0.7, edgecolor='black')
plt.bar(bar_positions + bar_width/2, percentages_spam, width=bar_width, color='red', label='Spam', alpha=0.7, edgecolor='black')

# Adicionando linhas verticais para os thresholds
plt.axvline(thresholds[0] * (len(bins) - 1), color='green', linestyle='--', label='< 30% Spam')
plt.axvline(thresholds[1] * (len(bins) - 1), color='green', linestyle='--', label='> 70% Spam')

# Adicionar rótulos para as áreas de corte
plt.text(thresholds[0] * (len(bins) - 1) / 2, plt.ylim()[1] * 0.85, '< 30%\n80% Pop', color='blue', fontsize=10, ha='center')
plt.text((thresholds[0] * (len(bins) - 1) + thresholds[1] * (len(bins) - 1)) / 2, plt.ylim()[1] * 0.85, '30-70%\nAnálise Manual\n12.5% Pop', color='black', fontsize=10, ha='center')
plt.text((thresholds[1] * (len(bins) - 1) + (len(bins) - 1)) / 2, plt.ylim()[1] * 0.85, '> 70%\n7.5% Pop', color='red', fontsize=10, ha='center')

# Adicionando rótulos e título
plt.title('Distribuição das probabilidades preditas com áreas de corte')
plt.xlabel('Probabilidade de ser spam')
plt.ylabel('Percentual da população')
plt.xticks(bar_positions, labels=[f'{int(b * 100)}%' for b in bins[:-1]])
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Mover legenda para fora do gráfico

# Adicionando percentuais no topo das barras
for x, y in zip(bar_positions, percentages_general):
    plt.text(x - bar_width/2, y + 1, f'{y:.1f}%', ha='center', va='bottom', fontsize=8)
for x, y in zip(bar_positions, percentages_spam):
    plt.text(x + bar_width/2, y + 1, f'{y:.1f}%', ha='center', va='bottom', fontsize=8, color='red')

# Ajustar o layout para evitar sobreposição
plt.subplots_adjust(right=0.75)  # Deixa espaço para a legenda

# Mostrar o gráfico
plt.show()