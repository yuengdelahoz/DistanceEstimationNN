from Network.Net import Network

n = Network()
n.initialize('topology_04')
n.train()
n.evaluate()

n1 = Network()
n1.initialize('topology_03')
n1.train()
n1.evaluate()
