---
title: "KLダイバージェンスとFisher情報行列の関係"
date: 2025-05-21T22:41:00+09:00
draft: false
tags: ["KL Divergence", "Fisher's Information Matrix", "Taylor Expansion"]
summary: パラメトリックな確率モデルの微小変化に対するKLダイバージェンスはFisher情報行列を計量行列として計算できる
---

{{<colored_note color="red">}}
This note is unfinished. I will translate it into English soon.
{{</colored_note>}}

{{<colored_note color="blue">}}
KLダイバージェンスをTaylor展開すると2次の係数としてFisher情報行列が現れる
{{</colored_note>}}

パラメータ $\theta (\in \mathbb{R}^d)$で記述されるパラメトリックな確率モデル$p(x|\theta)$と，パラメータに$\delta\theta$の摂動が加わった$p(x|\theta+\delta\theta)$のKLダイバージェンス

$$
\begin{align*}
KL\left[p(x|\theta)\right \\| p(x|\theta + \delta \theta)] &= \int p(x|\theta)\log p(x|\theta) dx - \int p(x|\theta) \log p(x | \theta + \delta\theta)dx \\
\end{align*}
$$

について考える．

まず，Taylor展開
$$
\begin{align*}
            & \log p(x | \theta + \delta \theta) \\\\
    \approx & \log p(x | \theta) + \left(\pdv{\theta} \log p(x | \theta)\right)^\mathsf{T}\delta\theta + \frac{1}{2}\delta\theta^\mathsf{T} \left[\frac{\partial^2}{\partial \theta \partial \theta^\mathsf{T}}(\log p(x | \theta)) \right] \delta\theta
\end{align*}
$$
を利用して，
$$
\begin{align*}
        & KL\left[p(x|\theta)\right \\| p(x|\theta + \delta \theta)] \\\\
\approx & -\int p(x|\theta) \left(\pdv{\theta}\log  p(x|\theta)\right)^{\mathsf{T}} \delta\theta dx - \frac{1}{2} \delta\theta^\mathsf{T} \left[ \int p(x|\theta)\left(\frac{\partial^2}{\partial \theta \partial \theta^\mathsf{T}}(\log p(x | \theta)) \right) dx \right] \delta\theta
\end{align*}
$$
を得る．

ここで，第1項について，
$$
\begin{align*}
    (\text{1st-term}) &= -\int p(x|\theta) \left(\pdv{\theta}\log  p(x|\theta)\right)^{\mathsf{T}}  \delta\theta dx \\\\
    &= -\int p(x|\theta) \left(\frac{\pdv{\theta} p(x|\theta)}{p(x|\theta)}\right)^{\mathsf{T}}  \delta\theta dx \\\\
    &= -\int \left(\pdv{\theta} p(x|\theta)\right)^\mathsf{T}\delta\theta dx \\\\
    &= -\left[\pdv{\theta}\int p(x|\theta) dx \right]^\mathsf{T} \delta\theta \\\\
    &= -\left[\pdv{\theta}1 \right]^\mathsf{T} \delta\theta \\\\
    &= 0
\end{align*}
$$
である．

さらに，第2項について，
$$
\begin{align*}
& \left[ \frac{\partial^2}{\partial \theta \partial \theta^\mathsf{T}}(\log p(x | \theta)) \right]\_{i,j} \\\\
=& \frac{\partial^2}{\partial \theta\_{i} \partial \theta\_{j}}(\log p(x | \theta)) \\\\
=& \pdv{\theta\_i} \left( \frac{\pdv{\theta\_j}p(x|\theta)}{p(x|\theta)} \right) \\\\
=& \frac{1}{p(x|\theta)}\frac{\partial^2 p(x|\theta)}{\partial \theta\_{i} \partial \theta\_{j}} - \left(\frac{1}{p(x|\theta)}\frac{\partial p(x|\theta)}{\partial \theta\_i}\right)\left(\frac{1}{p(x|\theta)}\frac{\partial p(x|\theta)}{\partial \theta\_j}\right)
\end{align*}
$$
より，
$$
\begin{align*}
& \frac{\partial^2}{\partial \theta \partial \theta^\mathsf{T}}(\log p(x | \theta)) \\\\
=& \frac{1}{p(x|\theta)}\frac{\partial^2}{\partial \theta \partial \theta^{\mathsf{T}}}p(x|\theta) - \left(\pdv{\theta} \log p(x|\theta)\right)\left(\pdv{\theta} \log p(x|\theta)\right)^\mathsf{T}
\end{align*}
$$
となるので，
$$
\begin{align*}
& (\text{2nd-term}) \\\\
=& - \frac{1}{2} \delta\theta^\mathsf{T} \left[ \int p(x|\theta)\left(\frac{\partial^2}{\partial \theta \partial \theta^\mathsf{T}}(\log p(x | \theta)) \right) dx \right] \delta\theta \\\\
=& -\frac{1}{2}\delta\theta^\mathsf{T} \left[\int \left(\frac{\partial^2 p(x | \theta)}{\partial \theta \partial \theta^\mathsf{T}} \right) dx\right] \delta\theta \\\\
& + \frac{1}{2} \delta\theta^\mathsf{T} \left[ \int p(x|\theta) \left(\pdv{\theta} \log p(x|\theta)\right)\left(\pdv{\theta} \log p(x|\theta)\right)^\mathsf{T} dx \right] \delta\theta \\\\
=& -0 + \frac{1}{2}\delta\theta^{\mathsf{T}}\mathbb{E}_{x\sim p(x|\theta)}\left[ \left(\pdv{\theta} \log p(x|\theta)\right)\left(\pdv{\theta} \log p(x|\theta)\right)^\mathsf{T} \right]\delta\theta \\\\
=& \frac{1}{2}\delta\theta^{\mathsf{T}} \mathcal{I}(\theta) \delta\theta
\end{align*}
$$
となる．

以上より，
$$
\begin{equation}
KL\left[p(x|\theta)\right \\| p(x|\theta + \delta \theta)] \approx \frac{1}{2}\delta\theta^{\mathsf{T}} \mathcal{I}(\theta) \delta\theta
\end{equation}
$$
となることがわかる．
すなわち，パラメトリックな確率モデル$p(x|\theta)$において$\delta\theta$だけパラメータが微小変化した確率モデルについて，元のモデルとのKLダイバージェンスはFisher情報行列を計量として計算できる．
これに関連するのが情報幾何学におけるFisher計量である（はずだが，情報幾何学についてはまだ勉強できていない）．
