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

判断 $\nabla E(\mathbf{x}) \cdot \vec{\beta} \vec{\beta}$ 是否无旋：$\nabla \times (\nabla E(\mathbf{x}) \cdot \vec{\beta} \vec{\beta}) = \nabla (\nabla E(\mathbf{x}) \cdot \vec{\beta}) \times \vec{\beta} = \vec{\beta} \cdot \nabla (\nabla E(\mathbf{x})) \times \vec{\beta}$。弹簧的能量 $E = \sum \frac{1}{2} k (||\mathbf{x} - \mathbf{y}|| - L)^2$，$\nabla E = \sum k (\mathbf{x} - \mathbf{y}) (1 - \frac{L} {||\mathbf{x} - \mathbf{y}||} )$，$\nabla (\nabla E) = \sum k (I - \frac{L} {||\mathbf{x} - \mathbf{y}||^3} (||\mathbf{x} - \mathbf{y}||^2 I - (\mathbf{x} - \mathbf{y}) (\mathbf{x} - \mathbf{y})^\top))$。又 $\vec{\beta} \cdot I \times \vec{\beta} = \vec{\beta} \times \vec{\beta} = \vec{0}$。因此 $\vec{\beta} \cdot \nabla (\nabla E(\mathbf{x})) \times \vec{\beta} = \sum kL \vec{\beta} \cdot \frac{(\mathbf{x} - \mathbf{y}) (\mathbf{x} - \mathbf{y})^T}{||\mathbf{x} - \mathbf{y}||^3} \times \vec{\beta}$ 不一定为 0。故不一定能使用原有的最优化方法求解。


$$
\min_{\mathbf{x}} \quad g(\mathbf{x}) = \frac{1}{2 h^2}(\mathbf{x} - \mathbf{y})^\top   \mathbf{M} (\mathbf{x} - \mathbf{y}) + E(\mathbf{x}) \tag{5}
$$