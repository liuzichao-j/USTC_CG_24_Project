# 地面系的动力学方程

$\vec{F}_{\parallel} = \gamma^3 m \vec{a}_{\parallel}$

$\vec{F}_{\perp} = \gamma m \vec{a}_{\perp}$

$\vec{a} = \vec{a}_{\parallel} + \vec{a}_{\perp} = \frac{\vec{F} \cdot \vec{\beta}} {\beta^2} \vec{\beta} \frac{1} {\gamma^3 m} + (\vec{F} - \frac{\vec{F} \cdot \vec{\beta}} {\beta^2} \vec{\beta}) \frac{1} {\gamma m} = \frac{1} {\gamma m} (\vec{F} - \vec{F} \cdot \vec{\beta} \vec{\beta}) $

# 半隐式方法

$\vec{v}_{n+1} = \vec{v}_n + \vec{a} \Delta t = \vec{v}_n + h \cdot \frac{1} {\gamma} (\frac{\vec{F}} {m} - \frac{\vec{F}} {m} \cdot \vec{\beta} \vec{\beta})$

# 隐式方法

$$
\begin{align}
\mathbf{x}^{n+1} &= \mathbf{x}^{n} + h \mathbf{v}^{n+1} \\
\mathbf{v}^{n+1} &=\mathbf{v}^{n} + h \mathbf{M}^{-1} (\mathbf{f} _ {\text{int}}(\mathbf{x}^{n+1}) + \mathbf{f}_{\text{ext}} ) \cdot (I - \vec{\beta} \vec{\beta}^\top)
\end{align} \tag{3}
$$
$$
\mathbf{x}^{n+1} = \mathbf{x}^{n} + h \mathbf{v}^{n} + h^2 \mathbf{M}^{-1} (-\nabla E(\mathbf{x^{n+1}}) + \mathbf{f}_{\text{ext}} ) \cdot (I - \vec{\beta} \vec{\beta}^\top) \tag{4}
$$

定义 $\mathbf{y} := \mathbf{x}^n + h \mathbf{v}^n + h^2 \mathbf{M}^{-1} \mathbf{f}_{\text{ext}} \cdot (I - \vec{\beta} \vec{\beta}^\top)$，

$$
\frac{1}{h^2} \mathbf{M} (\mathbf{x}^{n+1} - \mathbf{y}) + \nabla E(\mathbf{x}^{n+1}) \cdot (I - \vec{\beta} \vec{\beta}^\top) = \mathbf{0}
$$

$I - \vec{\beta} \vec{\beta}^\top$ 与坐标无关（此处采用 $\vec{v}_n$ 计算，并非严格隐式，否则非线性，难以处理），不参与 $\nabla$ 运算。记 $\mathbf{x} = \mathbf{x}^{n+1} \in \mathbf{R}^{3n \times 1}$，

$$
\min_{\mathbf{x}} \quad g(\mathbf{x}) = \frac{1}{2 h^2}(\mathbf{x} - \mathbf{y})^\top   \mathbf{M} (\mathbf{x} - \mathbf{y}) + E(\mathbf{x}) \tag{5}
$$