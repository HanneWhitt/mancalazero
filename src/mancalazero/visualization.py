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
    pos=None,
    title=None,
    xlim=None,
    ylim=None,
    savefile=None
):

    G = draw_graph(nodes, edges)

    if pos == None:
        pos = get_node_positions(G)

    nx.draw_networkx(G, pos = pos)

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
        'pos': pos,
        'title': title,
        'xlim': xlim,
        'ylim': ylim
    }

    return properties


def MCTS_expansion_series(
    nodes_edges_series_list,
    savefolder=None
):
    
    # Get graph properties from last entry
    properties = MCTS_visualization(*nodes_edges_series_list[-1])

    # Create savefolder
    if not os.path.isdir(savefolder):
        os.mkdir(savefolder)

    # Draw the series
    for i, (nodes, edges) in enumerate(nodes_edges_series_list):
        title = f'Simulation_{i}'
        properties['title'] = title
        savefile = os.path.join(savefolder, title)
        MCTS_visualization(nodes, edges, **properties, savefile=savefile)


def draw_graph(
    nodes,
    edges
):
    G = nx.DiGraph()
    
    for node, node_description in nodes.items():
        G.add_node(node)
    
    for edge in edges:
        G.add_edge(*edge)

    return G


def get_node_positions(G):

    pos=graphviz_layout(G, prog='dot')

    return pos