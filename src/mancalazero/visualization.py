import os
try:
    from matplotlib import pyplot as plt
    import networkx as nx
    from networkx.drawing.nx_agraph import graphviz_layout
except ModuleNotFoundError:
    msg = """
        Use of MCTS visualization module also requires packages:
        
            - matplotlib >= 3.8.4
            - networkx >= 2.8.4
            - pygraphviz >= 1.9
        """
    raise ModuleNotFoundError(msg)


def MCTS_visualization(
    nodes,
    edges,
    node_label_keys='all',
    edge_label_keys='all',
    pos=None,
    title=None,
    xlim=None,
    ylim=None,
    figsize=(12, 7),
    font_size=8,
    node_size=4000,
    savefile=None,
):

    G = create_graph(nodes, edges)

    node_labels = create_labels(nodes, node_label_keys)
    edge_labels = create_labels(edges, edge_label_keys)

    if pos == None:
        pos = get_node_positions(G)

    plt.figure(figsize=figsize)

    nx.draw(
        G,
        pos = pos,
        node_size=node_size
    )
    nx.draw_networkx_labels(
        G,
        pos=pos,
        labels=node_labels,
        font_size=font_size    
    )
    nx.draw_networkx_edge_labels(
        G,
        pos=pos,
        edge_labels=edge_labels,
        font_size=font_size,
        rotate=False
    )

    plt.title(title)
    if xlim:
        plt.xlim(xlim)
    else:
        xlim = plt.xlim()

    if ylim:
        plt.ylim(ylim)
    else:
        ylim = plt.ylim()

    if savefile:
        plt.savefig(savefile, dpi = 300)

    plt.close()

    properties = {
        'node_label_keys': node_label_keys,
        'edge_label_keys': edge_label_keys,
        'pos': pos,
        'title': title,
        'xlim': xlim,
        'ylim': ylim,
        'figsize': figsize,
        'font_size': font_size,
        'node_size': node_size
    }

    return properties


def MCTS_expansion_series(
    nodes_edges_series_list,
    savefolder=None,
    **properties
):
    
    # Get graph properties from last entry
    properties = MCTS_visualization(*nodes_edges_series_list[-1], **properties)

    # Create savefolder
    if not os.path.isdir(savefolder):
        os.mkdir(savefolder)

    # Draw the series
    for i, (nodes, edges) in enumerate(nodes_edges_series_list):
        title = f'Simulation_{i}'
        print(title)
        properties['title'] = title
        savefile = os.path.join(savefolder, title)
        MCTS_visualization(nodes, edges, **properties, savefile=savefile)


def create_graph(
    nodes,
    edges
):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def create_label(
    description,
    keys='all',
    rounding=3
):
    if keys == 'all':
        keys = description.keys()
    label = []
    for k in keys:
        v = description[k]
        if isinstance(v, float):
            v = round(v, rounding)
        line = f'{k}: {v}'
        label.append(line)
    label = '\n'.join(label)
    return label


def create_labels(
    id_desc_dict,
    keys='all',
    rounding=3
):
    return {id: create_label(desc, keys, rounding) for id, desc in id_desc_dict.items()}


def get_node_positions(G):

    pos=graphviz_layout(G, prog='dot')

    return pos