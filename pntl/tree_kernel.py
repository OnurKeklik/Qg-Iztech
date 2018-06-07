import networkx as nx
from lxml import html
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from bs4.element import NavigableString
from networkx.algorithms.traversal.depth_first_search import dfs_tree
#from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math

def syntaxTreeToHtml(raw):
    newString = ''
    detect = 0
    tagStack = []
    currentDetection = False
    for i in range (0,len(raw)):
        if raw[i] == '(':
            if detect > 0 and len(tagStack) == detect and newString.strip()[-1] != '>':
                newString = newString + raw[i].replace(raw[i], "><")
            else:
                newString = newString + raw[i].replace(raw[i], "<")

            tagString = ''
            idd = i + 1
            for j in range (idd,len(raw)):
                if raw[j] == ' ' or raw[j] == '(' or raw[j] == ')':
                    tagStack.append(tagString)
                    break
                else:
                    tagString = tagString + raw[j]
            detect = detect + 1

        elif raw[i] == ')':
            if currentDetection == True:
                tagStack.pop()
                newString = newString + raw[i].replace(raw[i], ">")
            else:
                newString = newString + "</" + tagStack.pop() + raw[i].replace(raw[i], ">")
            detect = detect - 1
            currentDetection = False

        elif raw[i] == ' ':
            currentDetection = True

        elif currentDetection == False:
            newString = newString + raw[i]
    return newString

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def traverse(parent, graph, labels):
    labels[hash(parent)] = parent.name
    for node in parent.children:
        if isinstance(node, NavigableString):
            continue
        graph.add_edge(hash(parent), hash(node))
        traverse(node, graph, labels)

def calculateSynacticSimilarity(raw, raw2):
	#raw = "(S1(SBAR(WHADVP(WRB How))(S(NP(JJ many)(NNS ways))(VP(MD can)(VP(VB respond)(PP(TO to)(NP(DT the)(JJ green)(NN imperative)))))(. ?))))"
	#raw2 = "(S1(SBAR(WHADVP(WRB How))(S(NP(JJ many)(NNS ways))(VP(MD can)(VP(VB respond)(PP(TO to)(NP(DT the)(JJ green)(NN imperative)))))(. ?))))"

	raw = syntaxTreeToHtml(raw)
	raw2 = syntaxTreeToHtml(raw2)

	print(raw)
	print (raw2)
	#raw = "<html><head><title></title></head><body><p><p><br><p></body></html>"
	#raw2 = "<html><head><title></title></head><body><script></script><p></body></html>"

	soup = BeautifulSoup(raw, "lxml")
	html_tag = next(soup.children)        

	soup = BeautifulSoup(raw2, "lxml")
	html_tag2 = next(soup.children)     
	#H=G.subgraph(G.nodes()[0:2])  
	G = nx.DiGraph()
	H = nx.DiGraph()
	labels = {}     # needed to map from node to tag
	labels2={}
	#html_tag = html.document_fromstring(raw)
	traverse(html_tag, G, labels)
	traverse(html_tag2, H, labels2)
	"""
	from networkx.drawing.nx_agraph import graphviz_layout
	pos = graphviz_layout(G, prog='dot',args="-Gsize=10")
	#nx.draw(G,with_labels=False)

	label_props = {'size': 16,
				   'color': 'black',
				   'weight': 'bold',
				   'horizontalalignment': 'center',
				   'verticalalignment': 'center',
				   'clip_on': True}
	bbox_props = {'boxstyle': "round, pad=0.1",
				  'fc': "grey",
				  'ec': "b",
				  'lw': 1.5}


	nx.draw_networkx_edges(G, pos, arrows=False)
	ax = plt.gca()

	for node, label in labels.items():
			x, y = pos[node]
			ax.text(x, y, label,
					bbox=bbox_props,
					**label_props)

	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
	plt.show()
	"""


	subtrees_G=[]
	subtrees_H=[]

	for i,node in enumerate(G.nodes()):
		subtrees_G.append(dfs_tree(G,node))
		
	for i,node in enumerate(H.nodes()):
		subtrees_H.append(dfs_tree(H,node))


	def label_check(d1,d2):
		return d1['labels']==d2['labels']

	for subtree in subtrees_H:
		for node in subtree.nodes():
			subtree.node[node]['labels']=labels2[node]

	for subtree in subtrees_G:
		for node in subtree.nodes():
			subtree.node[node]['labels']=labels[node]

	all_subtrees=subtrees_G+subtrees_H

		
	v=[]
	w=[]
	for i,subtree in enumerate(all_subtrees):
		if subtree.nodes()!=[]:
			v.append(np.sum(np.array(list(map(lambda x: nx.is_isomorphic(subtree,x,node_match=label_check),subtrees_G)))))
			w.append(np.sum(np.array(list(map(lambda x: nx.is_isomorphic(subtree,x,node_match=label_check),subtrees_H)))))

	#print(v, w, cosine_similarity(v,w))
	return cosine_similarity(v,w)
