# MI201
Adversial attack of neural networks

```math
\frac{\delta loss(f(x, \omega), y(x))}{\delta x}

||x_{origin} - x_{noised}||_{\infty} \leq \epsilon = 4 / 255

z = net(x)
\delta = request_grad()
z_n = net(x + \delta)
obj = ||z - z_n||
```
norme infinie(Xorigin - Xmodifé) <= Espilon = 4/255

z = net(x)
créer variable delta : delta.require_grad()
zm = net (x+delta)
il faut que loss = norme(z-zm)² soit grand 
obj.backward pour savoir dans quel sens changer delta pour augmenter la loss

