import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nodes", help="assign nodes id for this run",default='0')
args = parser.parse_args()
nodes = args.nodes
f = open('out.txt','w')
f.write(str(nodes))
f.close()