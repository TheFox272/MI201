# MI201

Attaque de réseau de neurone apr ajout d'un bruit ciblé.

## Theorie 

```math
\begin{align}
&\frac{\delta loss(f(x, \omega), y(x))}{\delta x}\\
&||x_{origin} - x_{noised}||_{\infty} \leq \epsilon = 4 / 255\\
&z = net(x)\\
&z_n = net(x + \delta)\\
&objective = ||z - z_n||_2  ↗
\end{align}
```
créer variable delta : delta.request_grad() / .require_grad()\\
obj.backward pour savoir dans quel sens changer delta pour augmenter la loss


## Liens utiles

[Xie_Adversarial_Examples_for_ICCV](https://openaccess.thecvf.com/content_ICCV_2017/papers/Xie_Adversarial_Examples_for_ICCV_2017_paper.pdf)\\
[coursdeeplearningcolab](https://colab.research.google.com/github/achanhon/coursdeeplearningcolab/blob/master/Untitled19.ipynb#scrollTo=lTWt48SZwG0-)\\
[adversarial-attacks-with-fgsm-fast-gradient-sign-method](https://pyimagesearch.com/2021/03/01/adversarial-attacks-with-fgsm-fast-gradient-sign-method/)
