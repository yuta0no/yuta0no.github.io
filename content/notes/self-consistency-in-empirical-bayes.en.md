---
title: "指数分布族の自然パラメータに対する経験ベイズ推定値において成り立つ等式"
date: 2026-05-16T19:00:00+09:00
draft: false
tags: ["Empirical Bayes", "Exponential Family"]
summary: 十分統計量の期待値が事前分布と事後分布で一致する
---

{{<colored_note color="red">}}
This note is unfinished. I will translate it into English soon.
{{</colored_note>}}

モデルがハイパーパラメータ $\eta \in \mathbb{R}^k$ を持つ事前分布 $p(x \mid \eta)$ および尤度関数 $p(y \mid x)$ で与えられる場合について、ハイパーパラメータを経験ベイズ法によって推定することを考える。
ただし、事前分布は指数分布族で与えられるとする。
すなわち、

$$
\begin{equation*}
  p(x \mid \eta) = h(x) \exp(\eta^\top \mathbf{T}(x) - A(\eta))
\end{equation*}
$$

であるとする。

まず、$\int p(x \mid \eta) dx = 1$ より

$$
\begin{equation*}
  A(\eta) = \log \int h(x) \exp(\eta^\top \mathbf{T}(x)) dx
\end{equation*}
$$

であるから両辺微分して、

$$
\begin{align}
  \nabla_\eta A(\eta) &= \frac{\int h(x)\exp(\eta^\top \mathbf{T}(x)) \mathbf{T}(x)dx}{\int h(x)\exp(\eta^\top \mathbf{T}(x))dx} \nonumber \\\\
  &= \int h(x)\exp(\eta^\top \mathbf{T}(x) - A(\eta)) \mathbf{T}(x)dx \nonumber \\\\
  &= \mathbb{E}_{p(x\mid \eta)} \left[ \mathbf{T} \right]
\end{align}
$$

となる。

次に、経験ベイズ法では周辺尤度 $L(\eta) = p(y \mid \eta) = \int p(y \mid x) p(x \mid \eta) dx$ を $\eta$ について最大化するから、経験ベイズ推定値 $\hat{\eta}$ について、

$$
\begin{equation*}
  \left.\nabla_\eta L(\eta)\right|_{\eta=\hat{\eta}} = \mathbf{0}
\end{equation*}
$$

が成り立つ。
すなわち、

$$
\begin{equation*}
  \int p(y\mid x) h(x) \exp(\eta^\top \mathbf{T}(x) - A(\hat{\eta})) (\mathbf{T}(x) - \left.\nabla_\eta A(\eta)\right|_{\eta=\hat{\eta}}) dx = \mathbf{0}
\end{equation*}
$$

である。
ここに、式 (1) を代入すると、

$$
  \int p(y\mid x) h(x) \exp(\eta^\top \mathbf{T}(x) - A(\hat{\eta})) (\mathbf{T}(x) - \mathbb{E}\_{p(x\mid \hat{\eta})} \left[\mathbf{T}(x)\right]) dx = \mathbf{0}
$$

となる。
両辺を $\int p(y \mid x) p(x \mid \eta) dx$ で割って整理することで、以下の等式を得る。

$$
\mathbb{E}\_{p(x \mid y, \hat{\eta})} \left[ \mathbf{T}(x) \right] = \mathbb{E}_{p(x \mid \hat{\eta})} \left[ \mathbf{T}(x) \right]
$$

これは事前分布が指数分布族で与えられる場合における経験ベイズ推定値は、事前分布における十分統計量の期待値と事後分布における十分統計量の期待値を等しくさせるということを意味する。

このメモについては[^1]を参考にした。

[^1]: [伊庭幸人、「ベイズモデリングの世界」（岩波書店）](https://www.iwanami.co.jp/book/b341709.html)
