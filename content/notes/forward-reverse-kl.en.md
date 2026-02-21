---
title: "Forward KLとReverse KL"
date: 2026-02-21T23:00:00+09:00
draft: false
tags: ["KL Divergence", "ELBO", "MLE", "Knowledge Distillation"]
summary: KLダイバージェンスは非対称な性質を持つ
---

{{<colored_note color="red">}}
This note is unfinished. I will translate it into English soon.
{{</colored_note>}}


{{<colored_note color="blue">}}
Forward KLはmean-seekingでReverse KLはmode-seeking
{{</colored_note>}}

KLダイバージェンスはふたつの確率分布がどれだけ離れているかを測ることができる「距離」のようなものである（厳密な意味での距離ではないことに注意）．

たとえば一般の[変分推論](two-ways-of-elbo-derivation)においては，変分下限（ELBO）を最大化することは真の分布$P$と近似分布$Q$とのKLダイバージェンス
$$
D_\mathrm{KL}(q_\theta \\| p) = \int_{Z} q_\theta(z) \log \frac{q_\theta(z)}{p(z | x)} dz
$$
を最小化することと等価であり，ELBO最大化による$Q$の学習は$D_\mathrm{KL}(q_\theta \\| p)$の意味で確率分布$q_\theta(z)$を真の事後分布$p(z|x)$に近づけていると解釈できる．


KLダイバージェンスは非対称な関数であるため，一般の最適化問題において近似分布$Q$を真の分布$P$にKLダイバージェンスの意味で近づけたい際には
$$
\begin{equation}
D_\mathrm{KL}(P \\| Q) = \int_{X} P(x) \log \frac{P(x)}{Q(x)} dx
\end{equation}
$$
と
$$
\begin{equation}
D_\mathrm{KL}(Q \\| P) = \int_{X} Q(x) \log \frac{Q(x)}{P(x)} dx
\end{equation}
$$
の2つの選択肢が存在する．

一般に，式(1)をForward KLと呼び，式(2)をReverse KLと呼ぶ．
これら2つのKLダイバージェンスは以下に示すようにそれぞれ異なる性質を持つ．



## Forward KL

まず，Forward KLは$\log \frac{P(x)}{Q(x)}$の$P(x)$についての期待値である．
したがって，$P(x)$が比較的大きい値をとり，なおかつ$Q(x) = \epsilon$（ただし$\epsilon$は極めて小さな数）となるような区間$[x, x+\Delta x]$が存在する場合，$P(x) \log \frac{P(x)}{Q(x)}$は大きな値をとり，期待値を大きくする．

逆に，$P(x)=0$であるような区間においては$Q(x)$がどのような値をとっても期待値に影響しない．

つまり，Forward KLを小さくするためには，$P(x)$の確率密度が非零である領域全体をカバーするように$Q(x)$を動かすことが重要である (zero-avoiding)．


## Reverse KL

一方，Reverse KLは$\log \frac{Q(x)}{P(x)}$の$Q(x)$についての期待値である．

Forward KLとは異なり，$P(x)$が大きな値をとる領域においても$Q(x)$が小さい値であれば，KLの値に与える影響は小さい．
逆に，$P(x)$が小さい区間で$Q(x)$が大きい場合はKLが大きい値になる．
したがって，Reverse KLの最小化においては$P(x)$が小さい区間では$Q(x)$を小さくし，$P(x)$が大きい点に$Q(x)$の確率密度を集中させることが重要である (zero-forcing)．
なお，$P(x)$が大きい領域が複数存在する場合でも，そのすべてで$Q(x)$を大きくする必要はないことに注意が必要である．


## 最適化時の分布の動き方の例

以下では，ふたつのKLの違いを定性的に確認するために，簡単な設定での最適化問題を例として考える．

真の分布として
$$
P(x) = 0.5 \times \mathcal{N}\left(x; \begin{bmatrix} -4.0 \\\\ 0 \end{bmatrix}, I_2\right) + 0.5 \times \mathcal{N}\left(x; \begin{bmatrix} 4.0 \\\\ 0 \end{bmatrix}, I_2\right)
$$
という2つの峰をもつ混合ガウス分布を設定し，
$$
Q(x) = \mathcal{N}(x; \mu, \Sigma)
$$
で近似する場合を考える．

特に，$Q$の$\mu, \Sigma$を学習可能パラメータとし，Adamを用いた勾配降下法によってForward KLを最小化する場合とReverse KLを最小化する場合の分布$Q$の動き方を比較する．
具体的なハイパーパラメータ等は[付録](#appendix)のソースコードを参照されたい．

なお，この最適化においてはKLダイバージェンスを直接評価するために真の分布$P(x)$からのサンプルを取得したり，$\log P(x)$を評価したりしているが，現実的な問題設定においては$P(x)$が未知であるためそのような操作が難しいことに注意が必要である（変分推論ではこの問題を回避するためにKLを直接最小化する代わりにELBOを最大化しているとも言える）．


### Forward KL

以下の動画は，反復法により$Q$がどのように変化するかを表している．
青色の等高線は$P$を表し，赤色のバツ印および楕円はそれぞれ$Q$の平均と$2\sigma$区間を表す．

Forward KLの最小化では，$P$の2つの峰の中心に$Q$の平均が近づいていき，多峰分布全体を覆うように共分散を大きくしていく (mean-seeking)．
これは，Forward KLは峰の取りこぼしを嫌うためである．

{{< video src="/video/forward.mp4" >}}


### Reverse KL

Reverse KLの最小化では$P$の片方の峰だけに$Q$が近づいていき，他方の峰は無視されている (mode-seeking)．
峰の取りこぼしはReverse KLの値に影響しない一方で，$P(x)$が小さい領域で$Q(x)$が大きくなることを嫌うためである．

したがって，Reverse KLによって$Q$を最適化する場合には，分散を過小評価してしまう（over-confidentになる）傾向がある．

{{< video src="/video/reverse.mp4" >}}



## それぞれのKLの用途

ここまでForward KLとReverse KLが異なる性質を持つことを確認してきた．
以下ではそれらのKLが実際の最適化や機械学習においてどのように利用されているかを考える．

まずReverse KLについては，先述したように変分ベイズにおける最小化対象として自然と導出される．
Reverse KLの計算においては近似分布$Q$からのサンプルを用いて期待値が近似できるため，計算が扱いやすいという特長を持っている．
これは最適化において求めたい分布$P$からのサンプルを取得することは難しい一方で，近似分布$Q$については具体的な形がわかっているためである．

最尤推定によるモデル学習はForward KLの最小化と同一視できる．

最尤推定は学習対象のデータ分布$p_\mathrm{data}(x)$に対して尤度関数$\log q(x | \theta)$が最大となるような$\theta$を求める方法である．
すなわち
$$
\mathbb{E}\_{p_\mathrm{data}(x)} \left[ \log q(x | \theta) \right]
$$
を最大化することで，観測データをうまく説明するモデルパラメータ$\theta$を推定する．

ここで
$$
\mathbb{E}\_{p_\mathrm{data}(x)} \left[ \log q(x | \theta) \right] = \mathbb{E}\_{p_\mathrm{data}(x)} \left[ \log p_\mathrm{data}(x) \right] - D_\mathrm{KL}(p_\mathrm{data}(x) \\| q(x | \theta))
$$
であり，右辺第1項は$p_\mathrm{data}(x)$固定のもとでは定数であることから，左辺を最大化することは右辺第2項のForward KLを最小化することと等価であることがわかる．

他には知識蒸留においてもForward KLが用いられることが多い[^1]．
具体的には，教師モデルを$q_t(x)$，生徒モデルを$q_s(x)$として
$$
D_\mathrm{KL}(q_t \\| q_s)
$$
を損失の一部として利用する．
Forward KLの性質を考慮すると，分類モデルに対してこのような損失を利用して知識蒸留した場合には，教師モデルが少しでも確率質量を割り当てたクラスについては生徒モデルの予測確率も非零となるよう訓練されることが想定される．

なお，LLMの知識蒸留の文脈においては，Forward KLが分布の峰に着目し，Reverse KLが分布の裾に着目する傾向をもつという発見に基づき，Forward KLだけでなくReverse KLも適応的に利用する方法が提案されているようである[^2]．


## 付録

以下に本記事に掲載した動画を作成するためのpythonスクリプトを示す．

{{< gist Yuta0no 6be41d50d602e5cfcf71272e21c2bd9d>}}



## Read Also

- ["KL Divergence: Forward vs Reverse?," Agustinus Kristiadi](https://agustinus.kristia.de/blog/forward-reverse-kl/)

[^1]: [Hinton+., "Distilling the Knowledge in a Neural Network," NIPS 2014 Workshop](https://arxiv.org/abs/1503.02531)
[^2]: [Wu+, "Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models," COLING 2025](https://arxiv.org/abs/2404.02657)
