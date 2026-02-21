---
title: "変分ベイズにおける2通りのELBO導出について"
date: 2025-07-30T17:00:00+09:00
draft: false
tags: ["KL Divergence", "Variational Bayes", "ELBO"]
summary: ELBO導出においてJensen's inequalityを使うかどうか
---

{{<colored_note color="blue">}}
変分ベイズにおいてELBO（変分下限，変分下界）を導出する際にJensen's inequalityを使うかどうかで式展開や解釈のしやすさが異なる（導出されるELBOは同じ）
{{</colored_note>}}


# はじめに

{{<colored_note color="black">}}
以下では，$X$を観測，$Z$を潜在変数（モデルのパラメータも含む）とする．
{{</colored_note>}}

変分ベイズ法は，真の事後分布$p(Z|X)$を変分分布$q(Z)$で近似する方法である．
事後分布はベイズの定理より，
$$
\begin{align*}
p(Z|X) &= \frac{p(Z, X)}{p(X)} \\\\
       &= \frac{p(X|Z)p(Z)}{p(X)} \\\\
       &= \frac{p(X|Z)p(Z)}{\int p(X|Z)p(Z) dZ}
\end{align*}
$$
と，事前分布および尤度から計算することができるが，分母の積分が解析的に解けるとは限らず，真の事後分布が計算可能とは限らない．
そこで，事後分布を直接計算するのではなく，新たな確率分布$q(Z)$を導入し，これを事後分布に近づけることを試みるのが変分ベイズである．

一般に，変分ベイズにおいては，まずはじめに周辺尤度（エビデンス）$p(X)$を利用してELBO（変分下限，変分下界）と呼ばれる項を導出する．
そして，そのELBOを最大化することで$q(Z)$を$p(Z|X)$に近づけていく（どのような意味で近づけていくかは後述する）．
このELBOの導出においては，[Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality)を使う場合と使わない場合の2つに大別できる．
インターネット上でよく見かけるのは前者であるが，個人的には後者の方が解釈しやすくなおかつエレガントであると思っている．
そこで，今回はこれらの方法を具体的に記述し，比較する．


# Jensen's inequalityを使う場合

まず，Jensen's inequalityを使ってELBOを導出する場合についてであるが，大まかな流れとしてはJensen's inequalityを使える形に周辺尤度を変形し，その後にJensen's inequalityを適用する．

Jensen's inequalityを使える形にするためには，以下のようにすればよい．

$$
\begin{align*}
  & \log p(X) \\\\
= & \log \left[ \int p(X, Z) \\, dZ\right] \\\\
= & \log \left[\int q(Z) \frac{p(X, Z)}{q(Z)}\\, dZ\right] \\\\
\end{align*}
$$

そして，対数関数は上に凸であることとJensen's inequalityを利用して，

$$
\begin{align*}
  & \log p(X) \\\\
= & \log \left[\int q(Z) \frac{p(X, Z)}{q(Z)}\\, dZ\right] \\\\
\geq & \int q(Z) \log \left[\frac{p(X, Z)}{q(Z)}\right] \\, dZ\\\\
=: & \ (ELBO)
\end{align*}
$$

とELBOを導出できる．

ここで問題になるのは，「なぜELBOを最大化することで$q(Z)$を$p(Z|X)$に近づけられるのか」という点である．
これを確認するためにもう1ステップ必要となる．

対数尤度とELBOの差を計算すると，

$$
\begin{align*}
& \log p(X) - (ELBO) \\\\
=& \int q(Z) \left[ \log p(X) - \log \frac{p(X, Z)}{q(Z)}\right] \\, dZ\\\\
=& \int q(Z) \left[ \log p(X) - \log \frac{p(Z | X)p(X)}{q(Z)}\right] \\, dZ\\\\
=& \int q(Z) \log \frac{q(Z)}{p(Z|X)} \\, dZ\\\\
=& KL\left[q(Z) \\| p(Z|X)\right]
\end{align*}
$$

となる．
今，観測$X$は固定値であることから，対数周辺尤度$\log p(X)$は定数であり，ELBOを大きくすることは$KL\left[q(Z) | p(Z|X)\right]$を小さくすることと同値である．
したがって，ELBO最大化により，[reverse-KL](forward-reverse-kl)の意味で変分分布を真の事後分布に近づけられるということがわかる．


# Jensen's inequalityを使わない場合

前節で示したように，Jensen's inequalityを利用してELBOを導出する場合，ELBOを導出しただけではELBO最大化が変分分布の最適化とどのような関係を持つのかが直観的には理解しづらい．
本節では，Jensen's inequalityを利用せずにELBOを導出すると，ELBO最大化とKLダイバージェンス最小化の関係が直観的に理解できるということを示す．

ここで必要となるのは，product ruleから導かれる$p(X) = \frac{p(Z, X)}{p(Z | X)}$とKLダイバージェンスが常に非負であるという知識だけである．

$$
\begin{align*}
& \log p(X) \\\\
=& \log \frac{p(Z, X)}{p(Z|X)} \\\\
=& \int q(Z) \log \frac{p(Z, X)}{p(Z|X)} \\, dZ \\\\
=& \int q(Z) \log \frac{p(Z, X)}{q(Z)} \\,dZ + \int q(Z) \log \frac{q(Z)}{p(Z|X)} \\,dZ \\\\
=& \ (ELBO) + KL[q(Z) \\| p(Z|X)]
\end{align*}
$$

なお，2行目から3行目の変形は，$p(X) = \frac{p(Z, X)}{p(Z|X)}$より$Z$に依存しないことおよび$q(Z)$が確率分布であること$\left(\int q(Z)\\, dZ = 1\right)$を利用した．

$p(X)$が定数であることとKLが非負であることから，この式展開においてはELBO最大化とKL最小化は同値であることが直観的に理解できる．


# おわりに

本ページでは，変分ベイズにおけるELBO導出の方法により，その解釈のしやすさがどう変わるかを確認した．
どちらの方法を用いても導出されるELBOは全く同じであるが，Jensen's inequalityを利用しない方法では，式展開の過程において対数周辺尤度がKLとELBOの和であることが明示的に現れる．

Jensen's inequalityを利用しない方法では，product ruleを利用したやや技巧的な式変形が必要となるが，導出される式が美しく，直観的にELBO最大化の意味が理解できるため個人的には好みである．

## Geminiによる比較

最後におまけとして，Geminiにこれら2つの式変形を比較してもらった．

おおむね私見と一致するが，GeminiはJensen's inequalityを使ったほうがエレガントだと感じるらしい．

|    | **Jensen's inequalityを利用する** | **Jensen's inequalityを利用しない** |
| -: | :- | :- |
| **強み** | なぜ「下限」なのかを数学的に簡潔に示す | ELBOとKLダイバージェンスの関係性を直感的に示す |
| **弱み** | ELBOとエビデンスの差が何か不明 | 「下限」であることが自明ではない |
| **適した目的** | 数学的な正当性をエレガントに証明したい | なぜELBOを最大化するのか、その意味を教育的に説明したい |

