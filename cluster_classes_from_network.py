import pandas as pd
import matplotlib.pyplot as plt

def split_classes(dataframe, enterprise):
    df_temp = dataframe.loc[df['Ultimate Parent'] == enterprise]
    class_list = df_temp['DWPI Class'].tolist()
    class_temp = []
    for item in class_list:
        for c in item.split("|"):
            class_temp.append(c.replace(' ', ''))
    return tuple(class_temp)

def extract_clusters(map):
    clusters = [occurrence[2] for occurrence in map]
    return set(clusters)

def extract_enterprises(map):
    clusters = extract_clusters(map)
    result = []
    for cluster in clusters:
        ent_list = []
        for item in map:
            if item[2] == cluster:
                ent_list.append(item[1])
        result.append((cluster, tuple(ent_list)))
    return result

def extract_classes(clusters, dataframe):
    clusters_classes = []
    for cluster in clusters:
        classes = []
        for enterprise in cluster[1]:
            for c in split_classes(dataframe, enterprise):
                classes.append(c)
        clusters_classes.append((cluster[0], classes))
    return clusters_classes

def count_classes(clusters):
    output_list = []
    for (cluster, classes) in clusters:
        mydict = {}
        for c in classes:
            if c in mydict.keys():
                mydict[c] += 1
            else:
                mydict[c] = 1
        output_list.append((cluster, sorted(mydict.items(), key = lambda x:x[1], reverse = True)))
    return output_list

def count_best(clusters, number):
    for (cluster, classes) in clusters:
        print(f'\033[36m{cluster}: \033[0m{classes[0:number]}')
    print()

def show_pies(cluster_classes, number):
    for (cluster, all_classes) in cluster_classes:
        classes = all_classes[0:number]
        labels = []
        counts = []
        sizes = []
        for item in classes:
            labels.append(item[0])
            counts.append(int(item[1]))
        amount = sum(counts)
        for count in counts:
            sizes.append(count/amount)
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', labeldistance=None )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.title(f"Migliori {number} classi per il cluster {cluster}")
        plt.show()

if __name__ == "__main__":
    data = pd.read_csv('clean_patents_batteries.tsv', sep='\t', low_memory=False)
    df = data[data["DWPI Class"].str.contains('|'.join(['X21', 'X22']), case=False, na=False)]
    df = df[["Ultimate Parent", "DWPI Class", ]].dropna()
    #print(df)

    map = []
    network = []

    with open("cluster_map.txt") as file:
        content = file.readlines()
        occurrences = [item.replace("\t", ",").replace("\n", "") for item in content]
        for occurrence in occurrences[1:]:
            occurrence = occurrence.split(",")
            map.append((occurrence[0], occurrence[1], occurrence[4]))
        file.close()

    with open("network_edges.txt") as file:
        content = file.readlines()
        for item in content:
            network.append(item.replace("\t", ",").replace("\n", ""))
        file.close()

    #print(map)

    # ESTRAZIONE DI TUTTE LE AZIENDE ASSOCIATE AD OGNI CLUSTER
    clusters_enterprises = extract_enterprises(map)

    # ESTRAZIONE DI TUTTE LE CLASSI ASSOCIATE AD OGNI CLUSTER
    cluster_classes = count_classes(extract_classes(clusters_enterprises, df))

    # ANALIZZA LE TOP 10 CLASSI DI OGNI CLUSTER
    number = 10
    count_best(cluster_classes, number)
    show_pies(cluster_classes, number)



