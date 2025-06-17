# Recursive Clustering

`recursive-clustering` é uma biblioteca Python para clusterização recursiva baseada em KMeans, com suporte a restrições de tamanho mínimo e máximo por grupo. É ideal para contextos como geolocalização de clientes, planejamento logístico, campanhas de marketing e segmentação de territórios.

## Por que usar Recursive KMeans?

KMeans tradicional pode formar clusters muito desbalanceados. Já o `recursive-kmeans` resolve isso ao dividir recursivamente os dados em clusters menores, respeitando limites configuráveis de tamanho.

## Instalação

Você pode instalar via GitHub:

```bash
pip install git+https://github.com/aapinheiro/recursive_clustering
```

## Como funciona

O algoritmo segue uma abordagem top-down:

- Inicia com todos os dados em um cluster;  
- Divide recursivamente em dois usando KMeans;  
- Verifica se os subclusters estão dentro do tamanho mínimo/máximo;  
- Repete a divisão até que todos os clusters estejam dentro dos limites definidos.

## Exemplo de uso 
```python
from recursive_kmeans import RecursiveClustering

# Definindo o clusterizador
rc = RecursiveClustering(
    geoloc_columns=["latitude", "longitude"],
    vars_encode=["cliente_id"],
    min_cluster_size=5,
    max_cluster_size=15,
    encode=True
)

# Treinando com um DataFrame Pandas
rc.fit(df)

# Obtendo os clusters e seus centroides
clustered_df, centroids = rc.get_cluster_dataframe()
```

## Estrutura esperada do DataFrame
Seu DataFrame deve conter pelo menos as colunas geográficas indicadas, como:

| latitude | longitude | cliente\_id |
| -------- | --------- | ----------- |
| -23.55   | -46.63    | 12345       |

## Parâmetros principais
| Parâmetro          | Tipo | Descrição                                                    |
| ------------------ | ---- | ------------------------------------------------------------ |
| `geoloc_columns`   | list | Colunas usadas na clusterização (ex: \['lat', 'lon'])        |
| `vars_encode`      | list | Colunas a serem usadas no nome codificado do cluster         |
| `min_cluster_size` | int  | Tamanho mínimo dos clusters                                  |
| `max_cluster_size` | int  | Tamanho máximo dos clusters                                  |
| `encode`           | bool | Se `True`, gera string única por cluster (`cluster_encoded`) |


## Output gerado 
O método get_cluster_dataframe() retorna:  
DataFrame com os dados clusterizados, incluindo colunas cluster e cluster_encoded (se encode=True);  
np.ndarray com os centroides de cada grupo.  
