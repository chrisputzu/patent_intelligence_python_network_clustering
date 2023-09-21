import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
from tqdm import tqdm

# Definizione della classe Cluster serve per la creazione di cluster, comunità
class Cluster:
    def __init__(self, id):
        self.id = id # id del cluster
        self.nodes = [] # etichetta noti
        self.citations = 0 # riferimenti

    def __len__(self):
        return len(self.nodes)

    def add_node(self, node, in_degree):
        self.nodes.append(node)
        self.citations += in_degree

# Definizione della classe GraphProcessor che esegue tutti i calcoli
class GraphProcessor:
    def __init__(self, data):
        self.data = data
        self.G = nx.DiGraph()
        self.top_clusters = {}

    # Estrazione dei riferimenti citati
    def extract_cited_refs(self):
        # Filtra i brevetti appartenenti alle classi X21 e X22
        cl_x22_x21 = self.data[self.data["DWPI Class"].str.contains('|'.join(['X21', 'X22']), case=False, na=False)]
        val = cl_x22_x21["Ultimate Parent"].dropna() # Ultimate Parent
        return val

    # Analisi dei riferimenti citati
    def parse_cited_refs(self, cited_refs):
        refs = cited_refs.split("|")
        parsed_refs = [ref.strip() for ref in refs]
        return parsed_refs

    # Creazione del grafo
    def create_graph(self, parsed_cited_refs):
        # Aggiunge gli archi al grafo - tqdm progress par
        for cited_refs in tqdm(parsed_cited_refs, desc='Adding edges to the graph'):
            for i in range(len(cited_refs) - 1):
                if self.G.has_edge(cited_refs[i], cited_refs[i + 1]):
                    self.G[cited_refs[i]][cited_refs[i + 1]]['weight'] += 1
                else:
                    self.G.add_edge(cited_refs[i], cited_refs[i + 1], weight=1)

    # Regolazione le dimensioni dei nodi
    def adjust_node_sizes(self):
        total_weighted_degrees = [d for n, d in self.G.degree(weight='weight')]
        out_weighted_degrees = [d for n, d in self.G.out_degree(weight='weight')]
        in_weighted_degrees = [total - out for total, out in zip(total_weighted_degrees, out_weighted_degrees)]
        node_sizes = []
        with tqdm(total=len(in_weighted_degrees), desc="Calculating node sizes") as pbar:
            for d in in_weighted_degrees:
                node_size = (d + 1) * 50
                node_sizes.append(node_size)
                pbar.update(1)
        self.node_sizes = node_sizes

    # Identificazione delle comunità
    def detect_communities(self):
        # Utilizzo l'algoritmo di Louvain per trovare la partizione ottimale
        with tqdm(total=1, desc="Detecting communities") as pbar:
            partition = community_louvain.best_partition(self.G.to_undirected())
            pbar.update(1)
        return partition

    # Elaborazione della partizione e ottenimento dei cluster
    def process_partition(self, partition):
        clusters = {}
        with tqdm(total=len(partition), desc="Processing partition") as pbar:
            for node, community_id in partition.items():
                if community_id not in clusters:
                    clusters[community_id] = Cluster(community_id)
                in_degree = self.G.in_degree(nbunch=node)
                clusters[community_id].add_node(node, in_degree)
                pbar.update(1)
        return clusters

    # Metodo combinato per rilevare le comunità e elaborare la partizione
    def detect_communities_and_process_partition(self):
        with tqdm(total=2, desc="Detecting communities and processing partition") as pbar:
            partition = self.detect_communities()
            pbar.update(1)
            filtered_clusters = self.process_partition(partition)
            pbar.update(1)
        top_clusters = self.select_top_clusters(filtered_clusters)  # selezione 5 cluster (modificabile)
        self.top_clusters = top_clusters

    # Seleziona i migliori cluster
    def select_top_clusters(self, clusters, num_clusters=5):
        sorted_clusters = sorted(clusters.items(), key=lambda x: x[1].citations, reverse=True)
        top_clusters = dict(sorted_clusters[:num_clusters])
        return top_clusters

    # Disegna il grafo di un cluster
    def plot_cluster_graph(self, cluster_nodes, cluster_id):
        cluster_subgraph = self.G.subgraph(cluster_nodes)
        # Utilizza il layout Fruchterman-Reingold
        pos = nx.spring_layout(cluster_subgraph, k=0.1, iterations=10)
        plt.figure(figsize=(12, 12))
        nx.draw(cluster_subgraph, pos, node_size=100, with_labels=True, font_size=8, node_color=f"C{cluster_id}")
        plt.title(f"Cluster {cluster_id + 1}")
        plt.show()

    # Disegna tutti i migliori cluster in un unico grafico
    def plot_all_clusters(self):
        plt.figure(figsize=(12, 12))
        valid_colors = ["#ff0000", "#00ff00", "#0000ff", "#ffd700", "#8b4513", '#00ffff']
        # Utilizza il layout Fruchterman-Reingold
        pos = nx.spring_layout(self.G, seed=42)  # Usa un seed per avere una disposizione coerente
        for cluster_id, cluster in enumerate(self.top_clusters.values()):  # Utilizza enumerate per ottenere un indice univoco
            if not valid_colors:  # Se non ci sono più colori disponibili, interrompi il ciclo
                break

            cluster_subgraph = self.G.subgraph(cluster.nodes)
            cluster_color = valid_colors.pop(0)  # Rimuovi il primo colore dalla lista e assegnalo al cluster
            nx.draw(cluster_subgraph, pos, node_size=100, with_labels=True, font_size=8,
                    node_color=cluster_color, alpha=0.8)
        plt.title("All Clusters")
        plt.show()

    # Funzione di appoggio per restituire la lista delle etichette dei nodi utili oer Vosviewer
    def get_node_labels(self):
        return list(self.G.nodes)

    # Funzione per creare gli id dei nodi dalle etichette (Vosviewer)
    def create_node_id_mapping(self):
        node_id_mapping = {}
        node_labels = self.get_node_labels()
        for node_id, label in enumerate(node_labels):
            node_id_mapping[label] = node_id
        return node_id_mapping

    # Funzione per creare un file txt con i dati dei nodi e dei cluster (Vosviewer)
    def save_cluster_map(self, filename, attributes=None, score=None):
        if attributes is None:
            attributes = ['id', 'label', 'x', 'y', 'cluster', 'weight']
        if score is None:
            score = ['score']

        # Ottieni gli ID dei nodi e la mappatura dei nodi
        node_id_mapping = self.create_node_id_mapping()

        with open(filename, 'w') as file:
            file.write('\t'.join(attributes + score) + '\n')
            for cluster_id, cluster in self.top_clusters.items():
                for node_label in cluster.nodes:
                    node_id = node_id_mapping[node_label]
                    in_degree = self.G.in_degree(nbunch=node_label, weight='weight')
                    item_attributes = {
                        'id': node_id,
                        'label': node_label,
                        'x': 0,
                        'y': 0,
                        'cluster': cluster_id,
                        'weight': in_degree,
                        'score': cluster.citations / len(cluster)
                    }
                    file.write('\t'.join(str(item_attributes[attr]) for attr in attributes + score) + '\n')

    # Funzione per creare un file txt per coi i dati degli archi (Vosviewer)
    def save_network_edges(self, filename):
        node_id_mapping = self.create_node_id_mapping()
        with open(filename, 'w') as file:
            for cluster in self.top_clusters.values():
                cluster_subgraph = self.G.subgraph(cluster.nodes)
                for edge in cluster_subgraph.edges(data='weight'):
                    source_id = node_id_mapping[edge[0]]
                    target_id = node_id_mapping[edge[1]]
                    weight = edge[2]
                    if weight is None:
                        weight = 1
                    file.write(f"{source_id}\t{target_id}\t{weight}\n")

    # Funzione per stampare a video i cluser, id_cluster e i label dei nodi
    def print_cluster_nodes(self):
        for cluster_id, cluster in self.top_clusters.items():
            print(f"Cluster {cluster_id + 1}:")
            for node in cluster.nodes:
                print(node)
            print()

    def calculate_and_print_graph_metrics(self):
        # Calcola la lunghezza media del percorso più breve per i nodi raggiungibili
        def average_shortest_path_length_for_reachable_nodes(G):
            paths_lengths = []
            for node in G.nodes():
                sp_lengths = nx.single_source_dijkstra_path_length(G, node)
                for key, value in sp_lengths.items():
                    if value != float('inf'):
                        paths_lengths.append(value)
            return sum(paths_lengths) / len(paths_lengths)

        # Calcolo del grado medio del grafo
        def average_degree(G):
            degrees = dict(G.degree())
            return sum(degrees.values()) / len(degrees)

        # Calcolo della lunghezza media del percorso più lungo per i nodi raggiungibili
        def average_longest_path_length_for_reachable_nodes(G):
            paths_lengths = []
            for node in G.nodes():
                if G.degree(node) == 0:
                    continue
                sp_lengths = nx.single_source_dijkstra_path_length(G, node)
                longest_path = max(sp_lengths.values())
                if longest_path != float('inf'):
                    paths_lengths.append(longest_path)
            return sum(paths_lengths) / len(paths_lengths)

        # Stampa del grafo e delle sue informazioni:
        # 1) Calcolo numero nodi
        print("Number of nodes:", len(self.G.nodes()))
        # 2) Calcolo numero archi
        print("Number of edges:", len(self.G.edges()))
        # 3) Calcolo densità del grafo
        density = nx.density(self.G)
        print(f"Density: {density}")
        # 4) Calcolo del diametro del grafo
        # Calcola il sottografo indotto dal componente fortemente connesso più grande
        largest_scc = max(nx.strongly_connected_components(self.G), key=len)
        # Sottografo
        subgraph = self.G.subgraph(largest_scc)
        # Diametro
        diameter = nx.diameter(subgraph)
        print(f"Diameter: {diameter}")
        # 5) Calcolo della lunghezza media del percorso più breve
        average_shortest_path_length = average_shortest_path_length_for_reachable_nodes(self.G)
        print(f"Average shortest path length for reachable nodes: {average_shortest_path_length}")
        # 6) Calcolo del grado medio del grafo
        avg_degree = average_degree(self.G)
        print(f"Average degree: {avg_degree}")
        # 7) Calcolo della lunghezza media del percorso più lungo
        average_longest_path_length = average_longest_path_length_for_reachable_nodes(self.G)
        print(f"Average longest path length for reachable nodes: {average_longest_path_length}")
        # 8) Calcolo del coefficiente di clustering medio del grafo
        average_clustering_coefficient = nx.average_clustering(self.G)
        print(f"Average clustering coefficient: {average_clustering_coefficient}")
        # 9) Calcola il sottografo indotto dal componente fortemente connesso più grande
        largest_scc = max(nx.strongly_connected_components(self.G), key=len)
        subgraph = self.G.subgraph(largest_scc)
        print(f"Largest strongly connected component: {subgraph}")
        # 10) Calcolo della centralità di closeness dei nodi
        closeness_centrality = nx.closeness_centrality(self.G)
        central_nodes_closeness_centrality = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"Top 10 central nodes (closeness centrality): {central_nodes_closeness_centrality}")

    # Funzione di appoggio per eseguire le operazioni dello script
    def process_graph(self):
        # Estrae i riferimenti citati
        cited_refs = self.extract_cited_refs()
        # Analizza i riferimenti citati
        parsed_cited_refs = [self.parse_cited_refs(refs) for refs in tqdm(cited_refs, desc='Parsing cited refs')]
        self.create_graph(parsed_cited_refs)
        # Rileva le comunità nel grafo
        partition = self.detect_communities()
        # Elabora le partizioni per ottenere i cluster
        clusters = self.process_partition(partition)
        # Seleziona i cluster migliori
        top_clusters = self.select_top_clusters(clusters)
        # Regola le dimensioni dei nodi nel grafo
        self.adjust_node_sizes()
        # Disegna il grafo di ogni cluster
        for cluster_id, cluster in top_clusters.items():
            self.plot_cluster_graph(cluster.nodes, cluster_id)
        self.detect_communities_and_process_partition()
        # Disegna tutti i migliori cluster in un unico grafico 2d
        self.plot_all_clusters()
        # Salva la mappa dei cluster in un file .txt
        self.save_cluster_map("cluster_map.txt")
        # Salva gli archi del network in un file .txt
        self.save_network_edges("network_edges.txt")
        # Stampa a schermo i id_cluster e nodi che lo compongono
        self.print_cluster_nodes()
        # Stampa a video metriche grafo
        self.calculate_and_print_graph_metrics()


if __name__ == "__main__":
    data = pd.read_csv('clean_patents_batteries.tsv', sep='\t', low_memory=False)
    graph_processor = GraphProcessor(data)
    graph_processor.process_graph()

