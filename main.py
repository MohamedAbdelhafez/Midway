import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nodes", help="assign nodes id for this run",default='0')
args = parser.parse_args()
nodes = str(args.nodes)
start_node = int(nodes[7:10])
end_node = int(nodes[11:14])
for ii in range (end_node - start_node):
    f = open('out.txt','w')
    f.write(str(start_node+ii) + " ")
    f.close()