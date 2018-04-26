import pydot
#required graphviz package installed

with open('data/classes.txt') as f:
    classes = [tuple(map(str, i.split(' ', 1))) for i in f]

with open('data/hierarchy.txt') as f:
    edges = [tuple(map(str, i.split(' ', 1))) for i in f]

dot = pydot.Dot(graph_type='graph', ratio="0.01")
dot.format = 'svg'

for i in classes:
    dot.add_node(pydot.Node(i[0], label=i[1]))

for i in edges:
    dot.add_edge(pydot.Edge(i[1], i[0]))

dot.write_svg('assets/nbirds_hierarchy.svg')
