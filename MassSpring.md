# 地面系的动力学方程

$\vec{F}_{\parallel} = \gamma^3 m \vec{a}_{\parallel}$

$\vec{F}_{\perp} = \gamma m \vec{a}_{\perp}$

$\vec{a} = \vec{a}_{\parallel} + \vec{a}_{\perp} = \frac{\vec{F} \cdot \vec{\beta}} {\beta^2} \vec{\beta} \frac{1} {\gamma^3 m} + (\vec{F} - \frac{\vec{F} \cdot \vec{\beta}} {\beta^2} \vec{\beta}) \frac{1} {\gamma m} = \frac{1} {\gamma m} (\vec{F} - \vec{F} \cdot \vec{\beta} \vec{\beta}) $

# 半隐式方法

$\vec{v}_{n+1} = \vec{v}_n + \vec{a} \Delta t = \vec{v}_n + h \cdot \frac{1} {\gamma} (\frac{\vec{F}} {m} - \frac{\vec{F}} {m} \cdot \vec{\beta} \vec{\beta})$

# 隐式方法

$$ \mathbf{x}^{n+1} = \mathbf{x}^{n} + h \mathbf{v}^{n+1} $$

$$ \mathbf{v}^{n+1} =\mathbf{v}^{n} + \frac{h}{\gamma} (I - \vec{\beta} \vec{\beta}^\top) \cdot \mathbf{M}^{-1} (\mathbf{f} _ {\text{int}}(\mathbf{x}^{n+1}) + \mathbf{f}_{\text{ext}} ) $$

$$ \mathbf{x}^{n+1} = \mathbf{x}^{n} + h \mathbf{v}^{n} + \frac{h^2}{\gamma} (I - \vec{\beta} \vec{\beta}^\top) \cdot \mathbf{M}^{-1} (-\nabla E(\mathbf{x^{n+1}}) + \mathbf{f}_{\text{ext}} ) $$

定义 $\mathbf{y} := \mathbf{x}^n + h \mathbf{v}^n + \frac{h^2}{\gamma} (I - \vec{\beta} \vec{\beta}^\top) \cdot \mathbf{M}^{-1} \mathbf{f}_{\text{ext}} $（预处理不同）

$$
\frac{\gamma}{h^2} \mathbf{M} (\mathbf{x}^{n+1} - \mathbf{y}) + (I - \vec{\beta} \vec{\beta}^\top) \cdot \nabla E(\mathbf{x}^{n+1}) = \mathbf{0}
$$

$I - \vec{\beta} \vec{\beta}^\top$ 与坐标无关（此处采用 $\vec{v}_n$ 计算，并非严格隐式，否则非线性，难以处理），不参与 $\nabla$ 运算。记 $\mathbf{x} = \mathbf{x}^{n+1} \in \mathbf{R}^{3n \times 1}$，

判断 $\nabla E(\mathbf{x}) \cdot \vec{\beta} \vec{\beta}$ 是否无旋：

$$\nabla \times (\nabla E(\mathbf{x}) \cdot \vec{\beta} \vec{\beta}) = \nabla (\nabla E(\mathbf{x}) \cdot \vec{\beta}) \times \vec{\beta} = \vec{\beta} \cdot \nabla (\nabla E(\mathbf{x})) \times \vec{\beta}$$

弹簧的能量 $E = \sum \frac{1}{2} k (||\mathbf{x} - \mathbf{y}|| - L)^2$，$\nabla E = \sum k (\mathbf{x} - \mathbf{y}) (1 - \frac{L} {||\mathbf{x} - \mathbf{y}||} )$
$$\nabla (\nabla E) = \sum k (I - \frac{L} {||\mathbf{x} - \mathbf{y}||^3} (||\mathbf{x} - \mathbf{y}||^2 I - (\mathbf{x} - \mathbf{y}) (\mathbf{x} - \mathbf{y})^\top))$$

又 $\vec{\beta} \cdot I \times \vec{\beta} = \vec{\beta} \times \vec{\beta} = \vec{0}$。因此
$$\vec{\beta} \cdot \nabla (\nabla E(\mathbf{x})) \times \vec{\beta} = \sum kL \vec{\beta} \cdot \frac{(\mathbf{x} - \mathbf{y}) (\mathbf{x} - \mathbf{y})^T}{||\mathbf{x} - \mathbf{y}||^3} \times \vec{\beta}$$
不一定为 0。原矢量方程无法转换为梯度方程，无法变为最小化能量函数的形式，故不能使用原有的最优化方法求解。

如果不管此处的不正确性，仍然继续按照此前的方式进行：$\min_{\mathbf{x}} g(\mathbf{x})$，其中
$$ \nabla g(\mathbf{x}) = \frac{\gamma}{h^2} \mathbf{M}(\mathbf{x} - \mathbf{y}) + (I - \vec{\beta} \vec{\beta}^\top) \nabla E(\mathbf{x}) $$

使用牛顿法：$ \mathbf{x}^{n+1} = \mathbf{x}^n - (\nabla^2 g)^{-1} \nabla \mathbf{g} $

$$ \nabla^2 g = \frac{\gamma}{h^2} \mathbf{M} I + (I - \vec{\beta} \vec{\beta}^\top) \nabla^2 E(\mathbf{x}) $$

一根弹簧能量的Hessian：

$$
\mathbf{H}_i = \nabla^2 E_i  
=k \frac{\mathbf{x}_i {\mathbf{x}_i}^\top}{\|\mathbf{x}_i\|^2}+k\left(1-\frac{L}{\|\mathbf{x}_i\|}\right)\left(\mathbf{I}-\frac{\mathbf{x}_i \mathbf{x}_i^{\mathrm{T}}}{\|\mathbf{x}_i\|^2}\right)
$$

计算方法相同，$ \nabla^2 g = \frac{\gamma}{h^2} \mathbf{M} + (I - \vec{\beta} \vec{\beta}^\top) \mathbf{H} $，注意 $I - \vec{\beta} \vec{\beta}^\top$ 在计算每一个顶点的时候都是不同的。

$\mathbf{x}^{n+1} = \mathbf{x}^n - (\nabla^2 g)^{-1} \nabla g (\mathbf{x}^n) = \mathbf{x}^n + \Delta \mathbf{x} $，方程组为：

$$ \nabla^2 g \Delta \mathbf{x} = -\nabla g $$

# 加速方法

此处需要使用 $g$ 的具体形式，而这是不能实现的。若按照步骤进行：

首先进行 Local Step：$ \mathbf{d}_i = L_i \frac{\mathbf{x}_i}{\|\mathbf{x}_i \|} $

再考虑 Global Step：也是梯度形式的
$$ \nabla g(\mathbf{x}) = \frac{\gamma}{h^2} \mathbf{M}(\mathbf{x} - \mathbf{y}) + (I - \vec{\beta} \vec{\beta}^\top) \nabla E(\mathbf{x}) $$

其中 $E = \frac{1}{2} \mathbf{x}^\top \mathbf{L}\mathbf{x} - \mathbf{x}^\top \mathbf{J} \mathbf{d} $，$\nabla E = \mathbf{L} \mathbf{x} - \mathbf{J} \mathbf{d}$

$$
\mathbf{L}=\left(\sum_{i=1}^s k_i \mathbf{A} _ i \mathbf{A} _ i^{\top}\right) \otimes \mathbf{I} _ 3, \quad \mathbf{J}=\left(\sum_{i=1}^s k_i \mathbf{A}_i \mathbf{S}_i^{\top}\right) \otimes \mathbf{I}_3
$$

$ (\mathbf{A}_i)_j = 1 (\text{i弹簧左端点为j}), -1 (\text{i弹簧右端点为j}), 0 (\text{其他}) $，$ (\mathbf{S}_i)_j = 1 (i = j), 0 (i \neq j) $

故求解的方程为

$$\gamma \mathbf{M}(\mathbf{x} - \mathbf{y}) + h^2 (I - \vec{\beta} \vec{\beta}^\top) (\mathbf{L} \mathbf{x} - \mathbf{J} \mathbf{d}) = 0 $$

$$(\gamma \mathbf{M} + h^2 (I - \vec{\beta} \vec{\beta}^\top) \mathbf{L}) \mathbf{x} = \gamma \mathbf{M} \mathbf{y} + h^2 (I - \vec{\beta} \vec{\beta}^\top) \mathbf{J} \mathbf{d} $$

固定点的处理：$\vec{x}^{'} = \mathbf{K} \vec{x}$，$\vec{x} = \mathbf{K}^{T} \vec{x}^{'} + \vec{b}$

$$ \mathbf{K} (\gamma \mathbf{M} + h^2 (I - \vec{\beta} \vec{\beta}^\top) \mathbf{L}) \mathbf{K}^{T} \vec{x}^{'} = \mathbf{K} (\gamma \mathbf{M} \vec{y} + h^2 (I - \vec{\beta} \vec{\beta}^\top) \mathbf{J} \mathbf{d}) - \mathbf{K} (\gamma \mathbf{M} + h^2 (I - \vec{\beta} \vec{\beta}^\top) \mathbf{L}) \vec{b}$$

注意到，该矩阵方程 $ \mathbf{A} \vec{x}^{'} = \vec{B} $，不再像经典情形 $ \mathbf{A} $ 是运动过程的不变量，无法使用该方法求解。考虑近似地能否适用，设可以表示为常矩阵乘上一个标量的形式（此时才能从式子中自由地提出和移动），$ \mathbf{A} = \mathbf{K} (\gamma \mathbf{M} + h^2 (I - \vec{\beta} \vec{\beta}^\top) \mathbf{L}) \mathbf{K}^{T} = \gamma \{ \mathbf{K} (\mathbf{M} + \frac{h^2}{\gamma} (I - \vec{\beta} \vec{\beta}^\top) \mathbf{L}) \mathbf{K}^{T} \} $。速度变大，$ \gamma \rightarrow \infty $，$ \mathbf{A} \approx \gamma \mathbf{K} \mathbf{M} \mathbf{K}^{T} $；而速度很小时，$ \gamma \rightarrow 0 $，$ \mathbf{A} \approx \mathbf{K} (\mathbf{M} + h^2 \mathbf{L}) \mathbf{K}^{T} $。可见这两种情形下 $ \mathbf{A} $ 之间的差异无法用常数刻画。仅在近似情形：$ m \gg k h^2 $，$ \mathbf{A} \approx \gamma \mathbf{K} \mathbf{M} \mathbf{K}^{T} $，此时可以使用该方法求解。鉴于实际仿真时常取 $ m = 1 / N, k = 1000, h = 0.01, k h^2 = 0.1 $，故该方法不适用。