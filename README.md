# MI201
Adversial attack of neural networks


norme infinie(Xorigin - Xmodifé) <= Espilon = 4/255

z = net(x)
créer variable delta : delta.require_grad()
zm = net (x+delta)
il faut que loss = norme(z-zm)² soit grand 
obj.backward pour savoir dans quel sens changer delta pour augmenter la loss